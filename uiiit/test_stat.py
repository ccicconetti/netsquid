__author__  = "Claudio Cicconetti"
__version__ = "0.1.0"
__license__ = "MIT"

import os
import unittest
import shutil
import time
from io import StringIO
from simstat import Conf, Stat, MultiStat
from utils import TestDirectory

def make_simple_stat(**kwargs):
    stat = Stat(Conf(par1=42, par2="hello"))
    for k, v in kwargs.items():
        stat.change_param(k, v)
    stat.add("mp", 0.1)
    stat.add("mp", -3.14)
    stat.add("mp", 100)
    stat.count("mc", 1)
    stat.count("mc", 2)
    return stat

class TestConf(unittest.TestCase):
    def test_match(self):
        conf = Conf(
            meaning_of_life=42,
            best_movie_ever='Gone with the wind',
            best_james_bond='Sean Connery')

        with self.assertRaises(KeyError):
            conf.match(best_actress_ever='Meryl Streep')

        with self.assertRaises(KeyError):
            conf.match(meaning_of_life=42, best_actress_ever='Meryl Streep')

        self.assertTrue(conf.match(meaning_of_life=42))

        self.assertTrue(conf.match(
            meaning_of_life=42, best_movie_ever='Gone with the wind'))

        self.assertFalse(conf.match(meaning_of_life="live happy"))

        self.assertFalse(conf.match(
            meaning_of_life=42, best_movie_ever='Ben-Hur'))

class TestStat(unittest.TestCase):

    def test_params(self):
        stat = Stat(Conf(par1="1", par2="0.5", par3="simple"))

        self.assertEqual("par1: 1, par2: 0.5, par3: simple", str(stat))

        stat.change_param('par1', 'mickey')
        self.assertEqual("par1: mickey, par2: 0.5, par3: simple", str(stat))

        stat.del_param('par1')
        self.assertEqual("par2: 0.5, par3: simple", str(stat))

    def test_eq(self):
        stat1 = make_simple_stat()

        stat2 = Stat(Conf(par2="hello", par1=42))
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
        self.assertEqual((5/4, 0), stat.get_avg_ci("m1"))

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
        self.assertEqual(0.5, stat.get_avg_ci("m1")[0])
        self.assertAlmostEqual(4.68443412, stat.get_avg_ci("m1")[1])
        self.assertEqual(values, stat.get_all("m1"))

        for _ in range(1000):
            stat.add("m1", 0.5)
        self.assertAlmostEqual(0.009971071490, stat.get_avg_ci("m1")[1])

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

    def test_merge(self):
        stat = Stat()

        values = [1, 2, 3]
        for v in values:
            stat.add("point-1", v)
        stat.add("point-2", 4)
        stat.merge("point-.*", "newpoint")

        self.assertEqual(4, len(stat.get_all("newpoint")))
        self.assertAlmostEqual(2.5, stat.get_avg("newpoint"))

        stat.count("count-1", 5)
        stat.count("count-1", 3)
        stat.count("count-2", 2)
        stat.count("count-3", -5)
        stat.merge("count-.*", "newcount")

        self.assertEqual(5, stat.get_all("newcount"))
        self.assertEqual(4, stat.get_count("newcount"))

        with self.assertRaises(KeyError):
            stat.merge(".*o.*", "newhybrid")

        with self.assertRaises(ValueError):
            stat.merge("point-.*", "count-1")

        stat.merge(".*does_not_exist.*", "newmetric")
        self.assertFalse("newmetric" in stat)

    def test_add_avg(self):
        stat = Stat()

        values = [1, 2, 3]
        for v in values:
            stat.add("point", v)
        stat.add_avg("point", "point-custom")
        stat.add_avg("point")

        self.assertAlmostEqual(2, stat.get_avg("point-custom"))
        self.assertAlmostEqual(2, stat.get_avg("point-avg"))

        stat.count("count", 5)
        stat.count("count", 3)
        stat.add_avg("count", "count-custom")
        stat.add_avg("count")

        self.assertAlmostEqual(4, stat.get_avg("count-custom"))
        self.assertAlmostEqual(1, stat.get_count("count-custom"))
        self.assertAlmostEqual(4, stat.get_avg("count-avg"))
        self.assertAlmostEqual(1, stat.get_count("count-avg"))

        with self.assertRaises(KeyError):
            stat.add_avg("does_not_exist")

        with self.assertRaises(ValueError):
            stat.add_avg("point", "count")

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

        self.assertTrue(os.path.exists(f'{path}/par1=42.par2=hello.mc.dat'))
        self.assertTrue(os.path.exists(f'{path}/par1=42.par2=hello.mp.dat'))

    @unittest.skip
    def test_print(self):
        stat = make_simple_stat()
        stat.print()

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

    def test_get_stats(self):
        mstat = MultiStat([
            make_simple_stat(par1=10, newpar='b', onlypar=0),
            make_simple_stat(par1=20, newpar='a'),
            make_simple_stat(par1=30, newpar='a'),
            ])
        
        self.assertEqual(3, len(mstat.get_stats(par2='hello')))
        self.assertEqual(2, len(mstat.get_stats(newpar='a')))
        self.assertEqual(1, len(mstat.get_stats(newpar='a', par1=20)))
        self.assertEqual(0, len(mstat.get_stats(newpar='a', par1=10)))

        with self.assertRaises(KeyError):
            mstat.get_stats(onlypar=0)

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

        self.assertTrue(os.path.exists(f'{path}/new_param=42.par1=42.par2=hello.mc.dat'))
        self.assertTrue(os.path.exists(f'{path}/new_param=42.par1=42.par2=hello.mp.dat'))
        self.assertTrue(os.path.exists(f'{path}/par1=42.par2=hello.mc.dat'))
        self.assertTrue(os.path.exists(f'{path}/par1=42.par2=hello.mp.dat'))

    def test_filter(self):
        mstat = MultiStat([
            make_simple_stat(par1=10),
            make_simple_stat(par1=20),
            make_simple_stat(par1=30),
            ])

        self.assertEqual(0, len(mstat.filter(par1=0)))
        self.assertEqual(1, len(mstat.filter(par1=10)))
        self.assertEqual(3, len(mstat.filter(par2="hello")))
        self.assertEqual(3, len(mstat.filter()))

    def test_param_values(self):
        mstat = MultiStat([
            make_simple_stat(par1=10),
            make_simple_stat(par1=20),
            make_simple_stat(par1=30),
            make_simple_stat(par1=30, par2="world"),
            make_simple_stat(par1=30, new_par="I'm new"),
            ])

        with self.assertRaises(KeyError):
            mstat.param_values('non-existing-par')
        
        self.assertEqual({10, 20, 30}, mstat.param_values('par1'))
        self.assertEqual({"hello", "world"}, mstat.param_values('par2'))

        with self.assertRaises(KeyError):
            mstat.param_values('new_par')

    def test_apply_to_all(self):
        mstat = MultiStat([
            make_simple_stat(par1=10),
            make_simple_stat(par1=20),
        ])

        conf1 = Stat(Conf(par1=10, par2="hello"))
        conf2 = Stat(Conf(par1=20, par2="hello"))

        avg = mstat[conf1].get_avg("mp")
        mstat.apply_to_all(lambda x : x.scale('mp', 2.0))
        self.assertAlmostEqual(avg * 2.0, mstat[conf1].get_avg("mp"))
        self.assertAlmostEqual(avg * 2.0, mstat[conf2].get_avg("mp"))

    def test_single_factor_data(self):
        mstat = MultiStat([
            make_simple_stat(par1=10),
            make_simple_stat(par1=20),
            make_simple_stat(par1=30),
        ])

        data = mstat.single_factor_data("par1")
        self.assertEqual({'mc', 'mp'}, data.keys())
        self.assertEqual({
            'par1=10.par2=hello',
            'par1=20.par2=hello',
            'par1=30.par2=hello'}, data['mc'].keys())
        self.assertEqual({
            'par1=10.par2=hello',
            'par1=20.par2=hello',
            'par1=30.par2=hello'}, data['mp'].keys())
        self.assertEqual(3, len(data['mc'].values()))
        self.assertEqual(3, len(data['mp'].values()))

    def test_single_factor_export(self):
        mstat = MultiStat([
            make_simple_stat(par1=10),
            make_simple_stat(par1=20),
            make_simple_stat(par1=30),
        ])

        with TestDirectory():
            mstat.single_factor_export('par1', 'test_directory')
            for par in range(10, 31, 10):
                self.assertTrue(os.path.isfile(
                    f'test_directory/par1={par}.par2=hello.mc.dat'))
                self.assertTrue(os.path.isfile(
                    f'test_directory/par1={par}.par2=hello.mp.dat'))

    @unittest.skip
    def test_save_to_json(self):
        mstat = MultiStat([
            make_simple_stat(),
            make_simple_stat(new_param=42),
            ])
        with open('mstat.json', 'w') as outfile:
            mstat.json_dump(outfile)

    @unittest.skip
    def test_print(self):
        mstat = MultiStat([
            make_simple_stat(),
            make_simple_stat(new_param=42),
            ])
        mstat.print()

if __name__ == '__main__':
    unittest.main()
