__author__  = "Claudio Cicconetti"
__version__ = "0.1.0"
__license__ = "MIT"

import unittest
import tempfile
from topology import Topology, EmptyTopology

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

    #
    # 0 ---> 1 ---> 2 <--> 3
    # ^                    |
    # +--------------------+
    #
    edges_test = [
            [0,  3],
            [1, 0],
            [2, 1],
            [2, 3],
            [3, 2],
    ]

    edges_test_sparse = [
            [0,  30],
            [10, 0],
            [20, 10],
            [20, 30],
            [30, 20],
    ]

    def test_invalid_type(self):
        with self.assertRaises(ValueError):
            Topology("unknown-topology")

    def test_chain(self):
        with self.assertRaises(EmptyTopology):
            Topology("chain", size=0)

        with self.assertRaises(ValueError):
            Topology("chain", size=-1)

        net1 = Topology("chain", size=1)
        self.assertEqual(1, net1.num_nodes)
        self.assertEqual(0, net1.diameter())

        net2 = Topology("chain", size=2)
        self.assertEqual(2, net2.num_nodes)
        self.assertEqual(1, net2.diameter())

        net5 = Topology("chain", size=5)
        self.assertEqual(5, net5.num_nodes)
        self.assertEqual(4, net5.diameter())

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
            [[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 4], [4, 3]],
            net5.edges())            

        self.assertEqual(
            [[0, 1], [1, 2], [2, 3], [3, 4]],
            net5.biedges())

    def test_ring(self):
        with self.assertRaises(EmptyTopology):
            Topology("ring", size=0)

        with self.assertRaises(ValueError):
            Topology("ring", size=-1)

        net1 = Topology("ring", size=1)
        self.assertEqual(1, net1.num_nodes)
        self.assertEqual(0, net1.diameter())

        net2 = Topology("ring", size=2)
        self.assertEqual(2, net2.num_nodes)
        self.assertEqual(1, net2.diameter())

        net5 = Topology("ring", size=5)
        self.assertEqual(5, net5.num_nodes)
        self.assertEqual(2, net5.diameter())
        self.assertEqual({1, 4}, net5.neigh(0))
        self.assertEqual({0, 2}, net5.neigh(1))
        self.assertEqual({1, 3}, net5.neigh(2))
        self.assertEqual({2, 4}, net5.neigh(3))
        self.assertEqual({3, 0}, net5.neigh(4))

    def test_grid(self):
        with self.assertRaises(EmptyTopology):
            Topology("grid", size=0)

        with self.assertRaises(ValueError):
            Topology("grid", size=-1)

        net1 = Topology("grid", size=1)
        self.assertEqual(1, net1.num_nodes)
        self.assertEqual("{'0': set()}", str(net1))
        self.assertEqual(0, net1.diameter())

        net4 = Topology("grid", size=4)
        self.assertEqual(16, net4.num_nodes)
        self.assertEqual(6, net4.diameter())

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
            self.assertEqual(2, net.diameter())

    def test_edges(self):
        with self.assertRaises(EmptyTopology):
            _ = Topology("edges")

        net = Topology("edges", edges=self.edges_test)

        self.assertEqual(4, net.num_nodes)

        self.assertEqual(3, net.diameter())

        self.assertEqual({3}, net.neigh(0))
        self.assertEqual({0}, net.neigh(1))
        self.assertEqual({1, 3}, net.neigh(2))
        self.assertEqual({2}, net.neigh(3))

        # shortest path tree to go to 3
        prev, dist = net.spt(3)
        self.assertEqual(3, dist[0]) # distance 0 -> 3 = 3 hops
        self.assertEqual(1, prev[0]) # next hop from 0 to 3 is 1
        # from 0 to 3 the path is 1 --> 2
        self.assertEqual([1, 2], Topology.traversing(prev, 0, 3)) 

    def test_edges_sparse(self):
        net = Topology("edges", edges=self.edges_test_sparse)

        self.assertEqual(4, net.num_nodes)

        self.assertEqual({30}, net.neigh(0))
        self.assertEqual({0}, net.neigh(10))
        self.assertEqual({10, 30}, net.neigh(20))
        self.assertEqual({20}, net.neigh(30))

    def test_names(self):
        net = Topology("chain", size=5)

        # Use name functions without assigning names

        self.assertEqual({'0', '1', '2', '3', '4'}, net.node_names)

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
            net.assign_names(dict())

        node_names = {
            0: 'A',
            1: 'B',
            2: 'C',
            3: 'D',
        }

        with self.assertRaises(ValueError):
            net.assign_names(node_names)

        node_names[4] = 'E'
        node_names[5] = 'F'

        with self.assertRaises(ValueError):
            net.assign_names(node_names)

        del node_names[5]

        net.assign_names(node_names)
            
        self.assertEqual({'A', 'B', 'C', 'D', 'E'}, net.node_names)

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
        node_names[0] = 'Z'
        net.assign_names(node_names)

        self.assertEqual({'Z', 'B', 'C', 'D', 'E'}, net.node_names)

        self.assertEqual('Z', net.get_name_by_id(0))
        self.assertEqual(0, net.get_id_by_name('Z'))

    def test_copy_names(self):
        net = Topology("chain", size=4)

        net.assign_names({0: "Mickey", 1: "Goofy", 2: "Donald", 3: "Minnie"})

        net2 = Topology("chain", size=3)

        net2.copy_names(net)

        self.assertEqual({'Donald', 'Goofy', 'Mickey'}, net2.node_names)

    def test_extract_bidirectional(self):
        net = Topology("edges", edges=[
            [0, 1],
            [0, 4],
            [1, 0],
            [1, 3],
            [2, 1],
            [3, 1],
            [3, 2],
            [3, 4],
            [4, 3]
        ])
        
        node_names = {
            0: 'A',
            1: 'B',
            2: 'C',
            3: 'D',
            4: 'E',
        }

        net.assign_names(node_names)

        net_bi = net.extract_bidirectional()

        self.assertEqual({1}, net_bi.neigh(0))
        self.assertEqual({0, 3}, net_bi.neigh(1))
        self.assertEqual({1, 4}, net_bi.neigh(3))
        self.assertEqual({3}, net_bi.neigh(4))

        self.assertEqual('A', net_bi.get_name_by_id(0))
        self.assertEqual('B', net_bi.get_name_by_id(1))
        self.assertEqual('D', net_bi.get_name_by_id(3))
        self.assertEqual('E', net_bi.get_name_by_id(4))

        self.assertEqual(0, net_bi.get_id_by_name('A'))
        self.assertEqual(1, net_bi.get_id_by_name('B'))
        self.assertEqual(3, net_bi.get_id_by_name('D'))
        self.assertEqual(4, net_bi.get_id_by_name('E'))

        with self.assertRaises(KeyError):
            _ = net_bi.get_name_by_id(2)

        with self.assertRaises(KeyError):
            _ = net_bi.get_id_by_name('C')

    def test_extract_bidirectional_empty(self):
        uni = Topology("edges", edges=[[0, 1], [2, 1]])

        with self.assertRaises(EmptyTopology):
            _ = uni.extract_bidirectional()

    def test_copy_names_bigger(self):
        full = Topology("grid", size=3)
        names_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
        names_dict = dict()
        for i in range(len(names_list)):
            names_dict[i] = names_list[i]
        full.assign_names(names_dict)

        uni = Topology("edges", edges=[
            [0, 1],
            [0, 3],
            [1, 0],
            [1, 2],
            [1, 4],
            [3, 0],
            [3, 4],
            [3, 6],
            [4, 1],
            [4, 3],
            [4, 5],
            [4, 7],
            [5, 2],
            [5, 4],
            [5, 8],
            [6, 3],
            [6, 7],
            [7, 4],
            [7, 6],
            [7, 8],
            [8, 7]
        ])

        uni.copy_names(full)

        self.assertEqual(9, uni.num_nodes)
        self.assertEqual(set(names_list), uni.node_names)

        reduced = uni.extract_bidirectional()

        self.assertEqual(8, reduced.num_nodes)
        names_list.remove('C')
        self.assertEqual(set(names_list), reduced.node_names)

    def test_nexthop_distance(self):
        net = Topology("edges", edges=self.edges_test)

        # Next hop

        self.assertEqual(None, net.next_hop(0, 0))
        self.assertEqual(1, net.next_hop(0, 1))
        self.assertEqual(1, net.next_hop(0, 2))
        self.assertEqual(1, net.next_hop(0, 3))

        self.assertEqual(2, net.next_hop(1, 0))
        self.assertEqual(None, net.next_hop(1, 1))
        self.assertEqual(2, net.next_hop(1, 2))
        self.assertEqual(2, net.next_hop(1, 3))

        self.assertEqual(3, net.next_hop(2, 0))
        self.assertEqual(3, net.next_hop(2, 1))
        self.assertEqual(None, net.next_hop(2, 2))
        self.assertEqual(3, net.next_hop(2, 3))

        self.assertEqual(0, net.next_hop(3, 0))
        self.assertEqual(0, net.next_hop(3, 1))
        self.assertEqual(2, net.next_hop(3, 2))
        self.assertEqual(None, net.next_hop(3, 3))

        # Distance

        self.assertEqual(0, net.distance(0, 0))
        self.assertEqual(1, net.distance(0, 1))
        self.assertEqual(2, net.distance(0, 2))
        self.assertEqual(3, net.distance(0, 3))

        self.assertEqual(3, net.distance(1, 0))
        self.assertEqual(0, net.distance(1, 1))
        self.assertEqual(1, net.distance(1, 2))
        self.assertEqual(2, net.distance(1, 3))

        self.assertEqual(2, net.distance(2, 0))
        self.assertEqual(3, net.distance(2, 1))
        self.assertEqual(0, net.distance(2, 2))
        self.assertEqual(1, net.distance(2, 3))

        self.assertEqual(1, net.distance(3, 0))
        self.assertEqual(2, net.distance(3, 1))
        self.assertEqual(1, net.distance(3, 2))
        self.assertEqual(0, net.distance(3, 3))

    def test_longest_path(self):
        self.assertEqual(7, Topology("chain", 8).longest_path())
        self.assertEqual(8, Topology("grid", 3).longest_path())
        self.assertEqual(15, Topology("grid", 4).longest_path())

        edges = []
        for i in range(0, 5):
            edges.append([i, i+1])
            edges.append([i+1, i])
        edges.append([0, 5])
        self.assertEqual(5, Topology("edges", edges=edges).longest_path())
        edges.append([5, 0])
        self.assertEqual(5, Topology("edges", edges=edges).longest_path())

    def test_farthest_nodes(self):
        # bi-directional ring
        net = Topology("ring", size=4)

        self.assertEqual({(2, 0), (0, 2), (1, 3), (3, 1)}, net.farthest_nodes())

        # uni-directional ring
        edges = []
        num_nodes = 4
        for i in range(num_nodes):
            prev = i - 1 if i > 0 else num_nodes - 1
            edges.append([i, prev])
        net_uni = Topology("edges", edges=edges)

        self.assertEqual({(0, 1), (1, 2), (2, 3), (3, 0)}, net_uni.farthest_nodes())

    @unittest.skip
    def test_graphviz(self):
        Topology("grid", size=4).save_dot("mygraph")

if __name__ == '__main__':
    unittest.main()