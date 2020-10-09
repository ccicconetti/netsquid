__author__  = "Claudio Cicconetti"
__version__ = "0.1.0"
__license__ = "MIT"

import unittest
from simstat import Stat

class TestStat(unittest.TestCase):
    def test_params(self):
        stat = Stat(par1="1", par2="0.5", par3="simple")

        self.assertEqual("par1: 1, par2: 0.5, par3: simple", str(stat))

    def test_counters(self):
        stat = Stat()

        stat.count("m1", 1)
        stat.count("m1", 2.5)
        stat.count("m1", 1)
        stat.count("m1", 0.5)

        self.assertEqual(5, stat.get_count_sum("m1"))
        self.assertEqual(5/4, stat.get_count_avg("m1"))

        with self.assertRaises(KeyError):
            stat.get_count_sum("m2")

        with self.assertRaises(KeyError):
            stat.get_count_avg("m2")

    def test_add(self):
        stat = Stat()

        values = [1, 2, -1.5, 0.5]
        for v in values:
            stat.add("m1", v)

        self.assertEqual(values, stat.get_add("m1"))

        with self.assertRaises(KeyError):
            stat.get_add("m2")

if __name__ == '__main__':
    unittest.main()
