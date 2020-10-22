__author__  = "Claudio Cicconetti"
__version__ = "0.1.0"
__license__ = "MIT"

import unittest
from simstat import Stat

class TestStat(unittest.TestCase):
    @staticmethod
    def make_simple_stat():
        stat = Stat(par1=42, par2="hello")
        stat.add("mp", 0.1)
        stat.add("mp", -3.14)
        stat.add("mp", 100)
        stat.count("mc", 1)
        stat.count("mc", 2)
        return stat

    def test_params(self):
        stat = Stat(par1="1", par2="0.5", par3="simple")

        self.assertEqual("par1: 1, par2: 0.5, par3: simple", str(stat))

        stat.change_param('par1', 'mickey')
        self.assertEqual("par1: mickey, par2: 0.5, par3: simple", str(stat))

        stat.del_param('par1')
        self.assertEqual("par2: 0.5, par3: simple", str(stat))

    def test_eq(self):
        stat1 = self.make_simple_stat()

        stat2 = Stat(par2="hello", par1=42)
        self.assertEqual(stat1, stat2)

        stat3 = self.make_simple_stat()
        stat3.change_param(key='par3', value='newvalue')
        self.assertNotEqual(stat1, stat3)

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

    def test_return_metrics(self):
        stat = Stat()

        self.assertEqual(set(), stat.count_metrics())
        self.assertEqual(set(), stat.point_metrics())

        stat.add("m1", 1)

        self.assertEqual(set(), stat.count_metrics())
        self.assertEqual({"m1"}, stat.point_metrics())

        stat.add("m2", 1)
        stat.count("m3", 1)
        stat.count("m4", 1)

        self.assertEqual({"m3", "m4"}, stat.count_metrics())
        self.assertEqual({"m1", "m2"}, stat.point_metrics())

    def test_scale(self):
        stat = Stat()
        values = [1, 2, -1.5, 0.5]
        for v in values:
            stat.add("m1", v)
        self.assertEqual([1, 2, -1.5, 0.5], stat.get_all("m1"))

        stat.scale("m1", 2)
        self.assertEqual([2, 4, -3, 1], stat.get_all("m1"))

        stat.count("m2", 1)
        stat.count("m2", 2)
        stat.count("m2", 3)
        self.assertEqual(6, stat.get_sum("m2"))
        self.assertEqual(3, stat.get_count("m2"))

        stat.scale("m2", 0.5)
        self.assertEqual(3, stat.get_sum("m2"))
        self.assertEqual(3, stat.get_count("m2"))

        with self.assertRaises(KeyError):
            stat.scale("m3", 1)

    def test_content_dump_load(self):
        stat1 = self.make_simple_stat()
        content1 = stat1.content_dump()

        stat2 = Stat.make_from_content(
            params=content1['params'],
            points=content1['points'],
            counts=content1['counts'])
        content2 = stat2.content_dump()

        self.assertEqual(content1, content2)  

if __name__ == '__main__':
    unittest.main()
