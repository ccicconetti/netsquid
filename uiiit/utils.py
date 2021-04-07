"""This module includes some general-purpose utility functions and classes
"""

import os
import shutil
import socket
import pickle
import logging
import time
from multiprocessing import Process, Queue
from queue import Empty

__all__ = [
    "ParallerRunner",
    "SocketCollector",
    "TestDirectory",
]


class SocketSender:
    """Sends a pickle-serialized object via TCP socket.

    Parameters
    ----------
    address : str
        The address of the server.
    port : int
        The TCP server's port.

    """

    def __init__(self, address, port):
        self._address = address
        self._port = port

    def send(self, obj):
        """Send the given object to the configured server.

        Note that the socket is opened in this call and closed immediately
        after the object has been sent.

        Parameters
        ----------
        obj
            The object to be sent. Anything pickle-serializable is OK, though
            its maximum (serialized) size must fit in an unsigned 32-bit int.

        Raises
        ------
        ValueError
            If the object is too big.
        """

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((self._address, self._port))

        msg = pickle.dumps(obj)
        if len(msg) >= 2 ** 32:
            raise ValueError(
                f"The serialization of the object is too big ({len(msg)} bytes)"
            )

        try:
            hdr = int.to_bytes(len(msg), 4, byteorder="big", signed=False)

            logging.info(
                f"Sending object of size ({len(msg)}) to {self._address}:{self._port}"
            )
            sent = s.send(hdr + msg)
            assert sent == (4 + len(msg))

        finally:
            s.close()


class SocketCollector:
    """Collects pickle-serialized objects sent via a TCP socket.

    Parameters
    ----------
    address : str
        The address where to bind the listening socket.
    port : int
        The port where to bind the listing socket.

    """

    def __init__(self, address, port):
        self._address = address
        self._port = port

    def collect(self, expected):
        """Collect a given number of objects, then quit.

        Parameters
        ----------
        expected : int
            The number of objects expected to be collected.

        """

        ret = []  # the list of objects returned

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind((self._address, self._port))
            s.listen()
            logging.info(
                (
                    f"Listening on {self._address}:{self._port}, "
                    f"waiting for {expected} objects"
                )
            )

            while len(ret) < expected:
                clientsocket, address = s.accept()
                logging.debug(f"Connection from {address} established")

                try:
                    while len(ret) < expected:
                        hdr = clientsocket.recv(4)
                        if not hdr:
                            clientsocket.close()
                            break

                        msg_len = int.from_bytes(hdr, byteorder="big", signed=False)

                        buf = b""
                        total = 0

                        while total != msg_len:
                            msg = clientsocket.recv(msg_len - total)
                            buf += msg
                            total += len(msg)

                        assert len(buf) == msg_len

                        logging.info(
                            f"object received, {expected - len(ret) - 1} to go"
                        )

                        ret.append(pickle.loads(buf))

                finally:
                    clientsocket.close()

        finally:
            s.close()

        return ret


class SocketParallerRunner:
    """Run a given function in parallel and return the list of their return values.

    Spawns a number of workers and synchronize their I/O via `Queue` and sockets.

    Parameters
    ----------
    address : str
        The address where to bind the listening TCP socket.
    port : int
        The port where to bind the listing TCP socket.

    """

    def __init__(self, address, port):
        self._address = address
        self._port = port

    def _sub_func(self, qin, func):
        """Single worker called by `run()`."""

        while True:
            try:
                args = qin.get_nowait()
            except Empty:
                return

            SocketSender(self._address, self._port).send(func(args))

    def run(self, nworkers, func, args):
        """Run a given function in parallel and return the list of their return values.

        Spawns a number of workers and synchronize their I/O via `Queue` and sockets.

        Parameters
        ----------
        nworkers : int
            The number of workers to spawn.
        func : lambda
            The function to call.
        args : list
            The list of arguments. The size of this list is the same as the number
            of executions of the function.

        Returns
        -------
        A list of items, one for each function invoked.

        Raises
        ------
        ValueError
            If the number of workers is smaller than 1.

        """

        if nworkers < 1:
            raise ValueError(f"Invalid number of workers: {nworkers}")

        qin = Queue()
        for arg in args:
            qin.put(arg)

        processes = []
        for _ in range(nworkers):
            p = Process(target=SocketParallerRunner._sub_func, args=(self, qin, func))
            p.start()
            processes.append(p)

        collector = SocketCollector(self._address, self._port)

        ret = collector.collect(expected=len(args))

        for p in processes:
            p.join()

        return ret


class ParallerRunner:
    @staticmethod
    def _sub_func(qin, qout, func):
        """Single worker called by `run()`."""

        while True:
            try:
                args = qin.get_nowait()
            except Empty:
                return

            qout.put(func(args))

    @staticmethod
    def run(nworkers, func, args):
        """Run a given function in parallel and return the list of their return values.

        Parameters
        ----------
        nworkers : int
            The number of workers to spawn.
        func : lambda
            The function to call.
        args : list
            The list of arguments. The size of this list is the same as the number
            of executions of the function.

        Returns
        -------
        A list of items, one for each function invoked.

        Raises
        ------
        ValueError
            If the number of workers is smaller than 1.

        """

        if nworkers < 1:
            raise ValueError(f"Invalid number of workers: {nworkers}")

        qin = Queue()
        for arg in args:
            qin.put(arg)
        qout = Queue()

        # assert qin.qsize() == len(args)

        processes = []
        for _ in range(nworkers):
            p = Process(target=ParallerRunner._sub_func, args=(qin, qout, func))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        ret = []
        while not qout.empty():
            ret.append(qout.get_nowait())

        return ret


class TestDirectory:
    """Create a directory for tests that is removed upon exiting the context."""

    def __init__(self):
        self._path = "test_directory"
        self._rmdir()
        os.mkdir(self._path)

    def __enter__(self):
        return self._path

    def __exit__(self, type, value, traceback):
        self._rmdir()
        pass

    def _rmdir(self):
        if os.path.exists(self._path):
            shutil.rmtree(self._path)


class Chronometer:
    """Logs the time required to execute instructions."""

    def __init__(self):
        self._start = time.monotonic()

    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):
        logging.debug(f"Elapsed time: {time.monotonic() - self._start}")
