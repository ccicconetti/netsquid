__author__  = "Claudio Cicconetti"
__version__ = "0.1.0"
__license__ = "MIT"

import unittest
from topology import Topology

class TestTopology(unittest.TestCase):
    def test_invalid_type(self):
        with self.assertRaises(ValueError):
            Topology("unknown-topology")

    def test_chain(self):
        with self.assertRaises(ValueError):
            Topology("chain", size=0)

        with self.assertRaises(ValueError):
            Topology("chain", size=-1)

        net1 = Topology("chain", size=1)
        self.assertEqual(1, net1.num_nodes)

        net2 = Topology("chain", size=2)
        self.assertEqual(2, net2.num_nodes)

        net5 = Topology("chain", size=5)
        self.assertEqual(5, net5.num_nodes)

        (prev, dist) = net5.spt(1)
        self.assertEqual({0: 1, 1: None, 2: 1, 3: 2, 4: 3}, prev)
        self.assertEqual({0: 1, 1: 0, 2: 1, 3: 2, 4: 3}, dist)

        self.assertEqual({1}, net5.neigh(0))
        self.assertEqual({0, 2}, net5.neigh(1))
        self.assertEqual({3}, net5.neigh(4))

        self.assertEqual([3], Topology.traversing(prev, 4, 2))

        with self.assertRaises(RuntimeError):
            net5.spt(999)

        with self.assertRaises(KeyError):
            Topology.traversing(prev, 4, 0)

        self.assertEqual("{0: {1}, 1: {0, 2}, 2: {1, 3}, 3: {2, 4}, 4: {3}}", str(net5))

    def test_grid(self):
        with self.assertRaises(ValueError):
            Topology("grid", size=0)

        with self.assertRaises(ValueError):
            Topology("grid", size=-1)

        net1 = Topology("grid", size=1)
        self.assertEqual(1, net1.num_nodes)
        self.assertEqual("{0: set()}", str(net1))

        net4 = Topology("grid", size=4)
        self.assertEqual(16, net4.num_nodes)

        other_nodes = list(range(16))
        for u in [0, 3, 12, 15]:
            self.assertEqual(2, len(net4.neigh(u)))
            other_nodes.remove(u)
        for u in [5, 6, 9, 10]:
            self.assertEqual(4, len(net4.neigh(u)))
            other_nodes.remove(u)
        for u in other_nodes:
            self.assertEqual(3, len(net4.neigh(u)))

        (prev, dist) = net4.spt(15)
        self.assertEqual(6, dist[0])
        self.assertEqual(5, len(Topology.traversing(prev, 0, 15)))

    @unittest.skip
    def test_graphviz(self):
        Topology("grid", size=4).save_dot("mygraph")

if __name__ == '__main__':
    unittest.main()