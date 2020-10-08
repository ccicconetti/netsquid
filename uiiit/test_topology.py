__author__  = "Claudio Cicconetti"
__version__ = "0.1.0"
__license__ = "MIT"

import unittest
import tempfile
from topology import Topology

class TestTopology(unittest.TestCase):
    brite_topo = \
b"""Topology: ( 20 Nodes, 37 Edges )
Model ( 2 ): 20 1000 100 1 2 1 10 1024

Nodes: (4)
0 209.00 89.00 9 9 -1 RT_NODE
1 689.00 291.00 3 3 -1 RT_NODE
2 323.00 292.00 11 11 -1 RT_NODE
3 614.00 744.00 3 3 -1 RT_NODE

Edges: (5):
0 0 1 520.77 1.74 10.00 -1 -1 E_RT U
1 0 2 520.77 1.74 10.00 -1 -1 E_RT U
2 1 2 520.77 1.74 10.00 -1 -1 E_RT U
3 1 3 520.77 1.74 10.00 -1 -1 E_RT U
4 2 3 520.77 1.74 10.00 -1 -1 E_RT U
"""

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

        self.assertEqual(
            "{0: {1}, 1: {0, 2}, 2: {1, 3}, 3: {2, 4}, 4: {3}}",
            str(net5))

        self.assertEqual(
            [[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 4], [4, 3]],
            net5.edges())            

        self.assertEqual(
            [[0, 1], [1, 2], [2, 3], [3, 4]],
            net5.biedges())

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

    def test_incoming_id(self):
        net = Topology("grid", size=3)

        self.assertEqual({1, 3, 5, 7}, net.neigh(4))
        self.assertEqual(0, net.incoming_id(4, 1))
        self.assertEqual(1, net.incoming_id(4, 3))
        self.assertEqual(2, net.incoming_id(4, 5))
        self.assertEqual(3, net.incoming_id(4, 7))

        self.assertEqual(1, net.neigh_from_id(4, 0))
        self.assertEqual(3, net.neigh_from_id(4, 1))
        self.assertEqual(5, net.neigh_from_id(4, 2))
        self.assertEqual(7, net.neigh_from_id(4, 3))

        self.assertEqual({5, 7}, net.neigh(8))
        self.assertEqual(0, net.incoming_id(8, 5))
        self.assertEqual(1, net.incoming_id(8, 7))

        self.assertEqual(5, net.neigh_from_id(8, 0))
        self.assertEqual(7, net.neigh_from_id(8, 1))

    def test_brite(self):
        with tempfile.NamedTemporaryFile() as fp:
            fp.write(self.brite_topo)
            fp.flush()
            net = Topology("brite", in_file_name=fp.name)
            self.assertEqual(4, net.num_nodes)
            self.assertEqual({1, 2}, net.neigh(0))
            self.assertEqual({0, 2, 3}, net.neigh(1))
            self.assertEqual({0, 1, 3}, net.neigh(2))
            self.assertEqual({1, 2}, net.neigh(3))

    def test_names(self):
        net = Topology("chain", size=5)

        # Use name functions without assigning names

        self.assertEqual('0', net.get_name_by_id(0))
        self.assertEqual('4', net.get_name_by_id(4))

        self.assertEqual(0, net.get_id_by_name('0'))
        self.assertEqual(4, net.get_id_by_name('4'))

        with self.assertRaises(ValueError):
            net.get_id_by_name('5')

        with self.assertRaises(ValueError):
            net.get_id_by_name('-1')

        # Assign names
        with self.assertRaises(ValueError):
            net.assign_names([])

        with self.assertRaises(ValueError):
            net.assign_names(['A', 'B', 'C', 'D'])

        with self.assertRaises(ValueError):
            net.assign_names(['A', 'B', 'C', 'D', 'E', 'F'])

        net.assign_names(['A', 'B', 'C', 'D', 'E'])

        self.assertEqual('A', net.get_name_by_id(0))
        self.assertEqual('B', net.get_name_by_id(1))
        self.assertEqual('C', net.get_name_by_id(2))
        self.assertEqual('D', net.get_name_by_id(3))
        self.assertEqual('E', net.get_name_by_id(4))

        self.assertEqual(0, net.get_id_by_name('A'))
        self.assertEqual(1, net.get_id_by_name('B'))
        self.assertEqual(2, net.get_id_by_name('C'))
        self.assertEqual(3, net.get_id_by_name('D'))
        self.assertEqual(4, net.get_id_by_name('E'))

        with self.assertRaises(KeyError):
            _ = net.get_name_by_id(5)

        with self.assertRaises(KeyError):
            _ = net.get_id_by_name('F')

        # Re-assign names
        net.assign_names(['F', 'G', 'H', 'I', 'J'])

        self.assertEqual('F', net.get_name_by_id(0))
        self.assertEqual(0, net.get_id_by_name('F'))

    @unittest.skip
    def test_graphviz(self):
        Topology("grid", size=4).save_dot("mygraph")

if __name__ == '__main__':
    unittest.main()