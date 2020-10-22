__author__  = "Claudio Cicconetti"
__version__ = "0.1.0"
__license__ = "MIT"

import unittest
import os
import multiprocessing
from time import sleep
from random import uniform

from utils import ParallerRunner

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

if __name__ == '__main__':
    unittest.main()