__author__  = "Claudio Cicconetti"
__version__ = "0.1.0"
__license__ = "MIT"

import unittest

import logging
import multiprocessing
import os
import pickle
import socket
from time import sleep
from random import uniform

from utils import ParallerRunner, SocketCollector, SocketSender

def my_func(arg):
    sleep(uniform(0, 0.2))
    return arg[0] + arg[1]

def func_get_pid(arg):
    sleep(1)
    return os.getpid()

class TestUtils(unittest.TestCase):
    def test_run_operations(self):
        args = []
        expected = set()
        for i in range(50):
            args.append([i, i+1])
            expected.add(2 * i + 1)
        ret = ParallerRunner.run(10, my_func, args)

        self.assertEqual(expected, set(ret))

    def test_run_getpid(self):
        nworkers = 10

        #
        # num workers > number of tasks
        #
        args = [None for _ in range(nworkers // 2)]
        ret = ParallerRunner.run(nworkers, func_get_pid, args)

        # check that each worker has its own PID
        self.assertEqual(nworkers // 2, len(set(ret)))

        #
        # num workers < number of tasks
        #
        args = [None for _ in range(nworkers * 2)]
        ret = ParallerRunner.run(nworkers, func_get_pid, args)

        # check that each task has been executed
        self.assertEqual(nworkers * 2, len(ret))

        # check that PIDs have been recycled
        # (might not be true on all operating systems)
        self.assertLessEqual(nworkers, len(set(ret)))

    @unittest.skip
    def test_socket_collector(self):
        logging.basicConfig(level=logging.INFO)
        collector = SocketCollector('localhost', 21001)

        num_expected = 10
        collector.collect(num_expected)

    @unittest.skip
    def test_socket_sender(self):
        logging.basicConfig(level=logging.INFO)
        sender = SocketSender('localhost', 21001)

        big_msg = [x for x in range(2**20)]
        
        for i in range(10):
            logging.info(f'Sending message #{i}')
            sender.send(big_msg)

    @unittest.skip
    def test_socket_server(self):
        expected = [x for x in range(100000)]
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        try:
            s.bind(('localhost', 21001))
            s.listen()

            while True:
                print(f"Waiting for a connection")
                clientsocket, address = s.accept()
                print(f"Connection from {address} has been established")

                while True:

                    hdr = clientsocket.recv(4)
                    if not hdr:
                        clientsocket.close()
                        break
                    
                    msg_len = int.from_bytes(hdr, byteorder='big', signed=False)
                    print(f'Received header of a message with size {msg_len}')

                    buf = b''
                    total = 0

                    while total != msg_len:
                        msg = clientsocket.recv(msg_len - total)
                        print(f'Received message chunk of size {len(msg)}')
                        buf += msg
                        total += len(msg)
                    
                    assert len(buf) == msg_len
                    print(f'Full message of size {  msg_len} received')

                    obj = pickle.loads(buf)
                    assert obj == expected

        finally:
            s.close()

    @unittest.skip
    def test_socket_client(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(('localhost', 21001))

        big_msg = [x for x in range(100000)]

        try:
            while True:

                big_msg_ser = pickle.dumps(big_msg)

                hdr = int.to_bytes(len(big_msg_ser), 4, byteorder='big', signed=False)

                print(f"sending message of size 4+{len(big_msg_ser)} bytes")
                sent = s.send(hdr + big_msg_ser)
                print(f"send {sent} bytes")

                sleep(1)

        finally:
            s.close()

if __name__ == '__main__':
    unittest.main()