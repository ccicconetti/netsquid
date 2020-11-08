__author__  = "Claudio Cicconetti"
__version__ = "0.1.0"
__license__ = "MIT"

import unittest

import os
import tempfile

from topology import Topology, EmptyTopology, TopographyDist, Topography2D
from utils import TestDirectory

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

        with self.assertRaises(KeyError):
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

    def test_isedge(self):
        net_grid = Topology("grid", size=3)
        self.assertTrue(net_grid.isedge(0, 1))
        self.assertTrue(net_grid.isedge(1, 0))
        self.assertTrue(net_grid.isedge(4, 5))
        self.assertTrue(net_grid.isedge(5, 4))
        self.assertFalse(net_grid.isedge(0, 2))
        self.assertFalse(net_grid.isedge(2, 0))

        with self.assertRaises(KeyError):
            net_grid.isedge(0, 10)
        self.assertFalse(net_grid.isedge(10, 0))

        net_chain = Topology("edges", edges=[[1, 0], [2, 1]])
        self.assertTrue(net_chain.isedge(0, 1))
        self.assertTrue(net_chain.isedge(1, 2))
        self.assertFalse(net_chain.isedge(1, 0))
        self.assertFalse(net_chain.isedge(2, 0))
        self.assertFalse(net_chain.isedge(2, 1))
        self.assertFalse(net_chain.isedge(0, 2))

    def test_nodes(self):
        net_grid = Topology("grid", size=3)
        self.assertEqual({0,1,2,3,4,5,6,7,8}, net_grid.nodes())

        net_chain = Topology("edges", edges=[[1, 0], [2, 1]])
        self.assertEqual({0,1,2}, net_chain.nodes())

    def test_degrees(self):
        net = Topology("grid", size=3)

        self.assertEqual(2, net.degree(0))
        self.assertEqual(3, net.degree(1))
        self.assertEqual(4, net.degree(4))

        with self.assertRaises(KeyError):
            net.degree(9)

        self.assertEqual(4, net.max_degree())
        self.assertEqual(2, net.min_degree())
        self.assertAlmostEqual((2*4+3*4+4)/9, net.avg_degree())

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

    def test_copy_weights(self):
        # 0 --> 1 --> 2
        edges = [[1, 0], [2, 1]]
        net1 = Topology("edges", edges=edges)
        net1.change_weight(0, 1, 10)
        net1.change_weight(1, 2, 100)

        net2 = Topology("edges", edges=edges)
        net2.copy_weights(net1)
        self.assertEqual(10, net2.weight(0, 1))
        self.assertEqual(100, net2.weight(1, 2))

        net3 = Topology("edges", edges=[[1, 0]])
        net3.copy_weights(net1)
        self.assertEqual(10, net3.weight(0, 1))
        with self.assertRaises(KeyError):
            net3.weight(1, 2)

        with self.assertRaises(KeyError):
            net1.copy_weights(net3)

        net4 = Topology("edges", edges=edges, default_weight=7)
        net5 = Topology("edges", edges=edges)
        net5.copy_weights(net4)
        self.assertEqual(14, net5.distance(0, 2))

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

    def test_extract_bidirectional_weights(self):
        # 0 <--> 1 <--> 2
        uni = Topology("edges", edges=[[1, 0], [0, 1], [1, 2], [2, 1]])
        uni.change_weight(0, 1, 10)
        uni.change_weight(1, 2, 100)
        self.assertEqual(110, uni.distance(0, 2))

        net_bi = uni.extract_bidirectional()
        self.assertEqual(110, net_bi.distance(0, 2))

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

    def test_all_paths(self):
        net_grid = Topology('grid', size=3)

        with self.assertRaises(KeyError):
            net_grid.all_paths(0, 9)
        self.assertEqual([
            [], [1, 4], [1, 2, 5, 4], [1, 2, 5, 8, 7, 4],
            [1, 2, 5, 8, 7, 6], [1, 4, 5, 8, 7, 6], [1, 4, 7, 6],
            [1, 2, 5, 4, 7, 6]], net_grid.all_paths(0, 3))

        self.assertEqual([[], [1, 4]], net_grid.all_paths(0, 3, 2))

        self.assertEqual([
            [], [1, 4], [1, 2, 5, 4], [1, 4, 7, 6]],
            net_grid.all_paths(0, 3, 4))

        # 0 --> 1 --> 2
        net_chain = Topology('edges', edges=[[1, 0], [2,1]])
        self.assertEqual([[1]], net_chain.all_paths(0, 2))
        self.assertEqual([], net_chain.all_paths(2, 0))

    def test_connected(self):
        self.assertTrue(Topology("grid", size=1).connected())
        self.assertTrue(Topology("grid", size=2).connected())
        self.assertTrue(Topology("grid", size=4).connected())
        self.assertTrue(Topology("chain", size=5).connected())

        self.assertFalse(Topology("edges", edges=[[1, 0]]).connected())
        self.assertFalse(
            Topology("edges", edges=[[1, 0], [0, 1], [2, 1]]).connected())
        self.assertFalse(
            Topology("edges", edges=[[1, 0], [0, 1], [3, 4], [4, 3]]).connected())

    def test_change_weight(self):
        net = Topology("edges", edges=self.edges_test, default_weight=42)

        for e in self.edges_test:
            self.assertEqual(42, net.weight(e[1], e[0]))

        with self.assertRaises(KeyError):
            net.weight(0, 3)

        net.change_weight(3, 0, 1)
        net.change_weight(0, 1, 2)

        self.assertEqual(1, net.weight(3, 0))
        self.assertEqual(2, net.weight(0, 1))

    def test_change_all_weights(self):
        net = Topology("edges", edges=self.edges_test, default_weight=42)
        for e,counter in zip(self.edges_test, range(len(self.edges_test))):
            self.assertEqual(42, net.weight(e[1], e[0]))
            net.change_weight(e[1], e[0], counter)

        net.change_all_weights(1)
        for e in self.edges_test:
            self.assertEqual(1, net.weight(e[1], e[0]))

    def test_spt_weights(self):
        #
        #    1 --> 2
        #    ^     | 
        #    |     v
        # 0 -+     3
        #    |     ^
        #    |     |
        #    +-----+
        #
        edges = [
            [1, 0],
            [2, 1],
            [3, 2],
            [3, 0]
        ]
        net = Topology("edges", edges=edges, default_weight=2)

        self.assertEqual(4, net.diameter())
        self.assertEqual(4, net.longest_path()) # should be 6?
        self.assertEqual(float('inf'), net.distance(1, 0))
        self.assertEqual(0, net.distance(0, 0))
        self.assertEqual(2, net.distance(0, 1))
        self.assertEqual(4, net.distance(0, 2))
        self.assertEqual(2, net.distance(0, 3))

        net.change_weight(1, 2, 100)

        self.assertEqual(102, net.diameter())
        self.assertEqual(102, net.longest_path()) # should be 6?
        self.assertEqual(0, net.distance(0, 0))
        self.assertEqual(2, net.distance(0, 1))
        self.assertEqual(102, net.distance(0, 2))
        self.assertEqual(2, net.distance(0, 3))

        for e in edges:
            net.change_weight(e[1], e[0], 1.5)

        self.assertEqual(3, net.diameter())
        self.assertEqual(3, net.longest_path()) # should be 4.5?
        self.assertEqual(0, net.distance(0, 0))
        self.assertEqual(1.5, net.distance(0, 1))
        self.assertEqual(3, net.distance(0, 2))
        self.assertEqual(1.5, net.distance(0, 3))

        net.change_weight(0, 3, 10)

        self.assertEqual(4.5, net.diameter())
        self.assertEqual(10, net.longest_path())
        self.assertEqual(4.5, net.distance(0, 3))

    def test_spt_external_weights(self):
        net = Topology('edges', edges=[
            [1, 0], [2, 1], [3, 2], [4, 3], [4, 5], [5, 0]
        ])
        (prev, dist) = net.spt(4)
        self.assertEqual(2, dist[0])
        self.assertEqual(5, prev[0])

        weights = {
            1: { 0: 1 },
            2: { 1: 1 },
            3: { 2: 1 },
            4: { 3: 1, 5: 1},
            5: { 0: 1}
        }
        (prev, dist) = net.spt(4, weights=weights)
        self.assertEqual(2, dist[0])
        self.assertEqual(5, prev[0])

        weights[1][0] = 5
        weights[5][0] = 10
        (prev, dist) = net.spt(4, weights=weights)
        self.assertEqual(8, dist[0])
        self.assertEqual(1, prev[0])

        (prev, dist) = net.spt(4, weights=weights, combine_op=max)
        self.assertEqual(5, dist[0])
        self.assertEqual(1, prev[0])

        weights[1][0] = 20
        (prev, dist) = net.spt(4, weights=weights, combine_op=max)
        self.assertEqual(10, dist[0])
        self.assertEqual(5, prev[0])

    def test_minmax(self):
        net = Topology('edges', edges=[
            [1, 0], [2, 1], [3, 2], [4, 3], [5, 4], [5, 6], [6, 0]
        ])
        other = Topology('edges', edges=net.edges() + [[5, 1], [5, 2], [5, 3]])

        self.assertEqual([6], net.minmax(5, 0, other))

        other.change_weight(1, 5, 0.9)
        self.assertEqual([6], net.minmax(5, 0, other))

        other.change_weight(2, 5, 0.9)
        self.assertEqual([6], net.minmax(5, 0, other))

        other.change_weight(3, 5, 0.9)
        self.assertEqual([1, 2, 3, 4], net.minmax(5, 0, other))

        other.change_weight(1, 5, 1.1)
        self.assertEqual([6], net.minmax(5, 0, other))

        other.change_weight(0, 6, 10)
        self.assertEqual([6], net.minmax(5, 0, other))

        other.change_weight(6, 5, 1.2)
        self.assertEqual([1, 2, 3, 4], net.minmax(5, 0, other))

        self.assertEqual(None, net.minmax(0, 5, other))
        self.assertEqual(None, net.minmax(0, 100, other))
        with self.assertRaises(KeyError):
            net.minmax(100, 5, other)

    def test_distance(self):
        net = Topology('grid', size=2, default_weight=2)

        self.assertEqual(2, net.distance(0, 1))
        self.assertEqual(4, net.distance(0, 3))

        net.change_weight(1, 0, 3)
        self.assertEqual(2, net.distance(0, 1))

        net.change_weight(0, 1, 3)
        self.assertEqual(3, net.distance(0, 1))

    def test_distance_path(self):
        net = Topology('grid', size=3)

        net.change_weight(0, 1, 2)
        net.change_weight(1, 2, 3)
        net.change_weight(2, 5, 4)
        net.change_weight(5, 8, 5)

        self.assertEqual(2, net.distance_path(0, 1, []))
        self.assertEqual(5, net.distance_path(0, 2, [1]))
        self.assertEqual(14, net.distance_path(0, 8, [1, 2, 5]))
        self.assertEqual(4, net.distance_path(0, 8, [3, 6, 7]))

        with self.assertRaises(KeyError):
            _ = net.distance_path(0, 8, [4])

    def test_contains(self):
        # 0 --> 1 --> 2
        net = Topology("edges", edges=[[1, 0], [2, 1]])
        self.assertTrue(0 in net)
        self.assertTrue(1 in net)
        self.assertTrue(2 in net)
        self.assertTrue(-1 not in net)
        self.assertTrue(4 not in net)

    @unittest.skip
    def test_graphviz(self):
        Topology("grid", size=4).save_dot("mygraph")

class TestTopographyDist(unittest.TestCase):
    def test_make_from_topology(self):
        net = Topology("ring", size=4)
        topo = TopographyDist.make_from_topology(net, 5, 10)

        edges = [[0, 1], [1, 2], [2, 3]]
        different_elems = set()
        for e in edges:
            self.assertGreaterEqual(topo.distance(e[0], e[1]), 5)
            self.assertLessEqual(topo.distance(e[0], e[1]), 10)
            self.assertEqual(topo.distance(e[0], e[1]),
                             topo.distance(e[1], e[0]))
            different_elems.add(topo.distance(e[0], e[1]))
            different_elems.add(topo.distance(e[1], e[0]))

        self.assertEqual(3, len(different_elems))

        with self.assertRaises(KeyError):
            topo.distance(0, 2)

        with self.assertRaises(KeyError):
            topo.distance(0, 10)

        with self.assertRaises(KeyError):
            topo.distance(20, 10)

    def test_update_topology(self):
        net = Topology('grid', size=2, default_weight=2)
        self.assertEqual(4, net.distance(0, 3))

        topo = TopographyDist.make_from_topology(net, 10, 10)
        topo.update_topology(net)
        self.assertEqual(20, net.distance(0, 3))

        # topography has nodes unknown to topology: OK
        topo_another = TopographyDist()
        for u in range(4):
            for v in range(u):
                topo_another.set_distance(u, v, 0.1)
        topo_another.set_distance(10, 11, 1)
        topo_another.update_topology(net)
        self.assertEqual(0.2, net.distance(0, 3))

        # topology has edges unknown to topography: NOK
        net_another = Topology('grid', size=3)
        with self.assertRaises(KeyError):
            topo_another.update_topology(net_another)

class TestTopography2D(unittest.TestCase):
    def test_invalid_ctor(self):
        with self.assertRaises(ValueError):
            Topography2D('notexisting')

    def test_disc(self):
        with self.assertRaises(ValueError):
            Topography2D('disc', nodes=-1)
        with self.assertRaises(ValueError):
            Topography2D('disc', nodes=2, size=-1)

        # threshold very high: no disconnected nodes
        topo_connected = Topography2D('disc', nodes=10, size=1, threshold=2)

        for u in range(10):
            for v in range(10):
                self.assertLessEqual(topo_connected.distance(u, v), 2)
                self.assertLessEqual(topo_connected.distance(u, v),
                                    topo_connected.distance(v, u))
                u_pos = topo_connected[u]
                v_pos = topo_connected[v]
                expected_dist_sq = \
                    (u_pos[0] - v_pos[0]) ** 2 + \
                    (u_pos[1] - v_pos[1]) ** 2
                self.assertAlmostEqual(expected_dist_sq,
                                       topo_connected.distance(u, v) ** 2)
        self.assertEqual(90, len(topo_connected.edges()))
        self.assertEqual(0, len(topo_connected.orphans()))

        with self.assertRaises(KeyError):
            topo_connected[-1]

        with self.assertRaises(KeyError):
            topo_connected[11]

        # threshold very low: all nodes are disconnected
        topo_disconnected = Topography2D('disc', nodes=10, size=1, threshold=0)
        self.assertEqual(0, len(topo_disconnected.edges()))
        self.assertEqual(set(list(range(10))), topo_disconnected.orphans())

        for u in range(10):
            for v in range(10):
                if u == v:
                    self.assertEqual(0, topo_disconnected.distance(u, v))
                else:
                    with self.assertRaises(KeyError):
                        topo_disconnected.distance(u, v)

        # intermediate threshold: some nodes may be disconnected
        topo_mix = Topography2D('disc', nodes=100, size=1, threshold=0.1)
        for e in topo_mix.edges():
            self.assertLessEqual(topo_mix.distance(e[0], e[1]), 0.1)

    def test_square(self):
        topo = Topography2D('square', nodes=100, size=1, threshold=1.41422)
        self.assertEqual(0, len(topo.orphans()))
        greater_than_one = []
        for e in topo.edges():
            self.assertLessEqual(topo.distance(e[0], e[1]), 1.41422)
            if topo.distance(e[0], e[1]) > 1:
                greater_than_one.append(e)
        self.assertGreater(len(greater_than_one), 0)

    def test_export(self):
        topo = Topography2D('disc', nodes=10, size=1, threshold=1)
        with TestDirectory() as testdir:
            topo.export(testdir + "/nodes.dat", testdir + "/edges.dat")
            self.assertTrue(os.path.exists(testdir + "/nodes.dat"))
            self.assertTrue(os.path.exists(testdir + "/edges.dat"))

if __name__ == '__main__':
    unittest.main()