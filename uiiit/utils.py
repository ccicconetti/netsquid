"""This module includes some general-purpose utility functions and classes
"""

from multiprocessing import Process, Queue
from queue import Empty

__all__ = [
    "ParallerRunner"
    ]

class ParallerRunner:
  @staticmethod
  def _sub_func(qin, qout, func):
    """Single worker called by `run_parallel`."""

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
