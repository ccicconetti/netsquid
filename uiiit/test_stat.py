__author__  = "Claudio Cicconetti"
__version__ = "0.1.0"
__license__ = "MIT"

import unittest
from simstat import Stat

class TestStat(unittest.TestCase):
    def test_params(self):
        stat = Stat(par1="1", par2="0.5", par3="simple")

        self.assertEqual("par1: 1, par2: 0.5, par3: simple", str(stat))

    def test_double_definition(self):
        stat = Stat()

        stat.count("m1", 1)
        stat.count("m1", 0.5)

        with self.assertRaises(KeyError):
            stat.add("m1", 1)

        stat.add("m2", 1)
        stat.add("m2", 0.5)

        with self.assertRaises(KeyError):
            stat.count("m2", 1)

    def test_counters(self):
        stat = Stat()

        stat.count("m1", 1)
        stat.count("m1", 2.5)
        stat.count("m1", 1)
        stat.count("m1", 0.5)

        self.assertEqual(5, stat.get_sum("m1"))
        self.assertEqual(5, stat.get_all("m1"))
        self.assertEqual(4, stat.get_count("m1"))
        self.assertEqual(5/4, stat.get_avg("m1"))

        with self.assertRaises(KeyError):
            stat.get_sum("m2")

        with self.assertRaises(KeyError):
            stat.get_avg("m2")

    def test_add(self):
        stat = Stat()

        values = [1, 2, -1.5, 0.5]
        for v in values:
            stat.add("m1", v)

        self.assertEqual(2, stat.get_sum("m1"))
        self.assertEqual(4, stat.get_count("m1"))
        self.assertEqual(0.5, stat.get_avg("m1"))

        self.assertEqual(values, stat.get_all("m1"))

        with self.assertRaises(KeyError):
            stat.get_all("m2")

if __name__ == '__main__':
    unittest.main()
