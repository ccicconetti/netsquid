__author__  = "Claudio Cicconetti"
__version__ = "0.1.0"
__license__ = "MIT"

import os
import unittest
import shutil
from io import StringIO
from simstat import Conf, Stat, MultiStat

def make_simple_stat(**kwargs):
    stat = Stat(par1=42, par2="hello")
    for k, v in kwargs.items():
        stat.change_param(k, v)
    stat.add("mp", 0.1)
    stat.add("mp", -3.14)
    stat.add("mp", 100)
    stat.count("mc", 1)
    stat.count("mc", 2)
    return stat

class TestStat(unittest.TestCase):

    def test_params(self):
        stat = Stat(par1="1", par2="0.5", par3="simple")

        self.assertEqual("par1: 1, par2: 0.5, par3: simple", str(stat))

        stat.change_param('par1', 'mickey')
        self.assertEqual("par1: mickey, par2: 0.5, par3: simple", str(stat))

        stat.del_param('par1')
        self.assertEqual("par2: 0.5, par3: simple", str(stat))

    def test_eq(self):
        stat1 = make_simple_stat()

        stat2 = Stat(par2="hello", par1=42)
        self.assertEqual(stat1, stat2)

        stat3 = make_simple_stat()
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
        stat1 = make_simple_stat()
        content1 = stat1.content_dump()

        stat2 = Stat.make_from_content(
            params=content1['conf'],
            points=content1['points'],
            counts=content1['counts'])
        content2 = stat2.content_dump()

        self.assertEqual(content1, content2)

    def test_export(self):
        path = 'test_directory'
        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)
        stat = make_simple_stat()
        stat.export(path)

        self.assertTrue(os.path.exists(f'{path}/par1=42.par2=hello-mc.dat'))
        self.assertTrue(os.path.exists(f'{path}/par1=42.par2=hello-mp.dat'))

class TestMultiStat(unittest.TestCase):
    def test_multistat_ctor(self):
        mstat_empty = MultiStat()
        self.assertEqual(0, len(mstat_empty.all_confs()))

        stats = []
        for i in range(10):
            stats.append(make_simple_stat(counter=i))
        
        mstat = MultiStat(stats)        
        self.assertEqual(10, len(mstat.all_confs()))

        for conf in stats:
            self.assertTrue(conf in mstat)

        self.assertFalse(Conf(nonexisting=42) in mstat)

    def test_multistat_add(self):
        mstat = MultiStat()
        stat1 = make_simple_stat()
        conf1 = Conf(**stat1.conf().all_params())

        with self.assertRaises(KeyError):
            mstat[conf1]

        self.assertEqual(0, len(mstat))

        # add new item
        self.assertTrue(mstat.add(stat1))
        self.assertEqual({'mc'}, mstat[conf1].count_metrics())
        self.assertEqual(1, len(mstat))

        # add it again
        self.assertFalse(mstat.add(stat1))
        self.assertEqual(1, len(mstat))

        # add a new one
        stat2 = make_simple_stat(new_param=42)
        conf2 = Conf(**stat2.conf().all_params())
        self.assertTrue(mstat.add(stat2))
        self.assertEqual({'mc'}, mstat[conf2].count_metrics())
        self.assertEqual(2, len(mstat))

    def test_multistat_add_multiple(self):
        mstat = MultiStat()

        stat1 = make_simple_stat(new_param=1)
        stat2 = make_simple_stat(new_param=2)
        stat3 = make_simple_stat(new_param=3)

        self.assertTrue(mstat.add([stat1, stat2]))
        self.assertEqual(2, len(mstat))

        self.assertTrue(mstat.add([stat1, stat3]))
        self.assertEqual(3, len(mstat))

        self.assertFalse(mstat.add([stat1, stat2, stat3]))
        self.assertEqual(3, len(mstat))

    def test_json(self):
        # create a collection
        mstat = MultiStat()
        mstat.add(make_simple_stat())
        mstat.add(make_simple_stat(new_param=42))
        self.assertEqual(2, len(mstat.all_confs()))

        # serialize it to JSON
        io = StringIO()
        mstat.json_dump(io)

        # deserialize it from JSON to a new collection
        io.seek(0)
        mstat_new = MultiStat.json_load(io)
        self.assertEqual(2, len(mstat_new.all_confs()))

    def test_json_file(self):
        path = 'test_directory'
        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir('test_directory')

        # create a collection
        mstat = MultiStat()
        self.assertFalse(mstat)
        mstat.add(make_simple_stat())
        self.assertTrue(mstat)
        mstat.add(make_simple_stat(new_param=42))

        # serialize to file
        mstat.json_dump_to_file(f'{path}/mstat.json')

        # deserialize from file
        mstat_new = MultiStat.json_load_from_file(f'{path}/mstat.json')

        self.assertEqual(len(mstat), len(mstat_new))

    def test_json_file_empty(self):
        mstat = MultiStat.json_load_from_file(f'doesnotexist.json')
        self.assertFalse(mstat)

    def test_export(self):
        path = 'test_directory'
        if os.path.exists(path):
            shutil.rmtree(path)

        mstat = MultiStat()
        mstat.add(make_simple_stat())
        mstat.add(make_simple_stat(new_param=42))
        mstat.export(path)

        self.assertTrue(os.path.exists(f'{path}/new_param=42.par1=42.par2=hello-mc.dat'))
        self.assertTrue(os.path.exists(f'{path}/new_param=42.par1=42.par2=hello-mp.dat'))
        self.assertTrue(os.path.exists(f'{path}/par1=42.par2=hello-mc.dat'))
        self.assertTrue(os.path.exists(f'{path}/par1=42.par2=hello-mp.dat'))

    @unittest.skip
    def test_save_to_json(self):
        mstat = MultiStat([
            make_simple_stat(),
            make_simple_stat(new_param=42),
            ])
        with open('mstat.json', 'w') as outfile:
            mstat.json_dump(outfile)

if __name__ == '__main__':
    unittest.main()
