"""This module specifies a class that models the topology of a network.
"""

import math
import subprocess
import random

__all__ = [
    "Topology",
    "Topography",
    "TopographyDist",
    "Topography2D",
    ]

class EmptyTopology(Exception):
    def __init__(self):
        self.message = "Empty graph"

class Topology:
    """Topology of a network of nodes.

    The data structured is designed with sparse graphs in mind.

    The default weight/cost of each edge is assigned in the ctor and can
    then be updated with the `change_weight()` method.
    
    Parameters
    ----------
    type : { 'chain', 'grid', 'ring', 'brite', 'edges' }
        Type of the network to create.
    size : int, optional
        The actual meaning depends on the network type: 
        with "grid", this is the number of nodes on any side of the square;
        with "chain" and "ring" it's the number of nodes.
        It is unused with other topology types.
    in_file_name : str, optional
        Name of the input file. Only used with a "brite" topology.
    edges : list, optional
        List of edges [dst -> src]. Only used with an "edges" topology.
    default_weight : float
        The weight/cost associated to each directed edge.

    Properties
    ----------
    num_nodes : int
        Number of nodes in the network.
    node_names : set
        Set of node names.

    Raises
    ------
    ValueError
        If the type is not supported.
    EmptyTopology
        If the graph would be empty.
    """
    def __init__(self, type, size=1, in_file_name="", edges=[], default_weight=1):
        # Lazy-initialized list of the (unidirectional) edges
        self._edges = None

        # Lazy-initialized list of pairs of nodes with a bi-directional connection
        self._biedges = None

        # Lazy-initialized data structure that assigns a numeric 0-based
        # identifier, unique for each node, to every incoming edge
        self._incoming_id = None

        # Lazy-initialized data structure for string representation
        self._str_repr = None

        # Lazy-initialized next-hop and distance matrices.
        self._nexthop_matrix = None
        self._distance_matrix = None

        # The graph stored as a dictionary where:
        # key: the node identifier (u)
        # value: all the nodes that have a directed edge towards u
        self._graph = dict()

        # The names optionally assigned to nodes
        self._names_by_id = None
        self._id_by_names = None

        # The weight matrix, initially empty.
        self._weight_matrix = None
        self._default_weight = default_weight

        # Create the graph
        if type == "chain" or type == "ring":
            if size == 0:
                raise EmptyTopology()
            elif size < 1:
                raise ValueError(f'Invalid size value for chain: {size}')
            self.num_nodes = size
            for u in range(size):
                if u == 0:
                    self._graph[u] = set([1])
                elif u == (size-1):
                    self._graph[u] = set([size - 2])
                else:
                    self._graph[u] = set([u-1, u+1])
            if type == "ring":
                self._graph[0].add(size-1)
                self._graph[size-1].add(0)

        elif type == "grid":
            if size == 0:
                raise EmptyTopology()
            elif size < 1:
                raise ValueError(f'Invalid size value for grid: {size}')
            self.num_nodes = size * size

            for u in range(self.num_nodes):
                neighbours = []
                if u % size != 0:
                    neighbours.append(u - 1)
                if u % size != (size - 1):
                    neighbours.append(u + 1)
                if u // size != 0:
                    neighbours.append(u - size)
                if u // size != (size - 1):
                    neighbours.append(u + size)
                self._graph[u] = set(neighbours)
            
        elif type == "brite":
            # Read from file
            with open(in_file_name, 'r') as in_file:
                edge_mode = False
                for line in in_file:
                    if 'Edges:' in line:
                        edge_mode = True
                        continue

                    if not edge_mode:
                        continue

                    (_, u, v) = line.split(' ')[0:3]
                    (u, v) = (int(u), int(v))
                    if u not in self._graph:
                        self._graph[u] = set()
                    self._graph[u].add(v)
                    if v not in self._graph:
                        self._graph[v] = set()
                    self._graph[v].add(u)

                if not self._graph:
                    raise EmptyTopology()

                self.num_nodes = len(self._graph)

        elif type == "edges":
            if not edges:
                raise EmptyTopology()

            for [u, v] in edges:
                if u not in self._graph:
                    self._graph[u] = set()
                if v not in self._graph:
                    self._graph[v] = set()
                self._graph[u].add(v)

            self.num_nodes = len(self._graph)

        else:
            raise ValueError(f'Invalid topology type: {type}')

        # Create the list of node names with identifiers, until `assign_names()`
        # is called
        self.node_names = set([str(x) for x in range(self.num_nodes)])

    def __contains__(self, node):
        """Return True if `node` exists in the graph."""

        if node in self._graph:
            return True
        for neigh in self._graph.values():
            if node in neigh:
                return True
        return False

    def isedge(self, src, dst):
        """Return True if `dst` has an incoming edge from `src`.
        
        Parameters
        ----------
        src : int
            Source node identifier.
        dst : int
            Destination node identifier.
        
        Returns
        -------
        bool
            True if `dst` has an incoming edge from `src`, False otherwise.
            If `src` does not exist in the graph, always return False

        Raises
        ------
        KeyError
            If `dst` does not exist in the graph.
            
        """

        return src in self._graph[dst]

    def nodes(self):
        """Return the node identifiers."""

        ret = set()
        for u, neigh in self._graph.items():
            ret.add(u)
            for v in neigh:
                ret.add(v)
        return ret

    def degree(self, node):
        """Return the degree of a node.

        The degree is the number of incoming edges it has.

        Parameters
        ----------
        node : int
            Node identifier for which we return the degree.

        Returns
        -------
        int
            Degree of `node`.
        
        """

        return len(self._graph[node])

    def max_degree(self):
        """Return the maximum degree of the graph."""

        cur_max = None
        for u in self._graph:
            cur_degree = self.degree(u)
            if cur_max is None or cur_max < cur_degree:
                cur_max = cur_degree
        assert cur_max is not None
        return cur_max

    def min_degree(self):
        """Return the minimum degree of the graph."""

        cur_min = None
        for u in self._graph:
            cur_degree = self.degree(u)
            if cur_min is None or cur_min > cur_degree:
                cur_min = cur_degree
        assert cur_min is not None
        return cur_min

    def avg_degree(self):
        """Return the average degree of the graph."""

        return len(self.edges()) / len(self._graph)

    def assign_names(self, node_names):
        """Assign names to nodes. Can be called multiple times.

        Parameters
        ----------
        node_names : dict
            Assign a name (str) to each identifier (int).

        Raises
        ------
        ValueError
            If not all nodes are assigned a name or if there are some names
            that do not correspond to a node identifier.
        
        """

        if set(self._graph.keys()) != set(node_names.keys()):
            raise ValueError(f"Invalid names provided")

        self._names_by_id = dict()
        self._id_by_names = dict()

        for node_id, node_name in node_names.items():
            self._names_by_id[node_id] = node_name
            self._id_by_names[node_name] = node_id

        self.node_names = set(node_names.values())

        # Reset the data structure for the string representation
        self._str_repr = None

    def copy_names(self, other):
        """Copy the names from another topology.

        Parameters
        ----------
        other : `Topology`
            The topology from which to re-use names.
        
        Raises
        ------
        KeyError
            If `other` does not define some of the nodes that are in the
            current `Topology` object.
        
        """

        node_names = dict()
        for u in self._graph.keys():
            node_names[u] = other.get_name_by_id(u)
        
        self.assign_names(node_names)

    def copy_weights(self, other):
        """Copy the weights from another topology.

        Parameters
        ----------
        other : `Topology`
            The topology from which to re-use names.
        
        Raises
        ------
        KeyError
            If `other` does not define some of the nodes that are in the
            current `Topology` object.
        
        """

        self._default_weight = other._default_weight
        for e in self.edges():
            weight = other.weight(e[1], e[0])
            if weight != self._default_weight:
                self.change_weight(e[1], e[0], weight)

    def get_name_by_id(self, node_id):
        """Return the name corresponding to the given identifier.

        If nodes do not have assigned names, then the identifier itself is
        returned as a string.

        Raises
        ------
        KeyError
            The `node_id` does not exist.

        """

        if self._names_by_id:
            return self._names_by_id[node_id]

        return str(node_id)

    def get_id_by_name(self, node_name):
        """Return the node identifier corresponding to the given name.

        If nodes do not have assigned names, then a conversion from `node_name`
        to string is attempted.

        Raises
        ------
        ValueError
            Node names have not been assigned and `node_name` cannot be
            converted to an int, or the conversion succeeds but the value is
            out of the possible range.
        KeyError
            The `node_name` does not exist.

        """

        if self._id_by_names:
            return self._id_by_names[node_name]

        node_id = int(node_name)
        if node_id >= self.num_nodes or node_id < 0:
            raise ValueError(f"Invalid node name: {node_name}")

        return node_id
    
    def edges(self):
        """Return the list of unidirectional edges"""

        # Lazy initialization
        if self._edges is not None:
            return self._edges

        self._edges = []
        for u, neigh in self._graph.items():
            for v in neigh:
                self._edges.append([u, v])

        self._edges.sort()
        return self._edges 

    def biedges(self):
        """Return the list of pairs of node which have a bidirectional connection.

        A given pair of node (u,v) is returned only once, i.e., the returned
        list does not contain duplicates. Also, it is sorted in lexycographic
        order, and each pair is also sorted, thus the function returns a
        stable dataset.
        """

        # Lazy initialization
        if self._biedges is not None:
            return self._biedges

        self._biedges = []
        for u, neigh in self._graph.items():
            for v in neigh:
                pair = [u ,v]
                pair.sort()
                if pair not in self._biedges:
                    self._biedges.append(pair)

        self._biedges.sort()
        return self._biedges

    def incoming_id(self, target_node, other_node):
        """Return the identifier associated to an incoming edge.

        Parameters
        ----------
        target_node : int
            Node for which the edge is incoming.
        other_node : int
            Neighbor of target_node for which we return the edge identifier.
        """

        self._create_incoming_id()
        return self._incoming_id[target_node][other_node]

    def neigh_from_id(self, target_node, edge_id):
        """Return the neighbor associated to given edge identifier for a node.

        Parameters
        ----------
        target_node : int
            Node for which the edge is incoming.
        edge_id : int
            Edge identifier.

        Raises
        -----
        IndexError
            If `target_node` does not have an edge with the given edge identifier

        """

        self._create_incoming_id()
        for v, cur_id in self._incoming_id[target_node].items():
            if cur_id == edge_id:
                return v

        raise IndexError(f"Cannot find edge identifier {edge_id} for node {target_node}")

    def __repr__(self):
        if self._str_repr is None:
            self._str_repr = dict()
            for u, neigh in self._graph.items():
                u_name = self.get_name_by_id(u)
                self._str_repr[u_name] = set()
                for v in neigh:
                    self._str_repr[u_name].add(self.get_name_by_id(v))

        return str(self._str_repr)

    def neigh(self, node):
        """Return the neighbours of the given node
        """

        return self._graph[node]

    def spt(self, dst, opposite_weight=False):
        """Compute the shortest-path tree to `dst`.

        Parameters
        ----------
        dst : int
            The identifier of the node for which to compute the SPT.
        opposite_weight : bool
            Take the opposite of the weights.
        
        Returns
        -------
        The previous hop and distances of all nodes to reach `src`. This is
        returned as a pair of dictionaries: the first one for every node u
        is the identifier of the node to which us has to send its commodity
        to reach `src` along a shortest path; the second one is the
        distance from every node u to `src` along that path.
        """

        Q = []
        dist = {}
        prev = {}
        for v in self._graph.keys():
            Q.append(v)
            dist[v] = float('inf')
            prev[v] = None

        if dst not in dist:
            raise KeyError(f'{dst} is not in {Q}')

        dist[dst] = 0

        while Q:
            u = None
            last_value = None
            for node in Q:
                value = dist[node]
                if u is None or value < last_value:
                    u = node
                    last_value = value
            Q.remove(u)

            for v in Q:
                if v not in self._graph[u]:
                    continue
                # v is in Q and a neighbor of u
                alt_dist_u_v = self.weight(v, u)
                if opposite_weight:
                    alt_dist_u_v *= -1.0
                alt = dist[u] + alt_dist_u_v
                if alt < dist[v]:
                    dist[v] = alt
                    prev[v] = u

        return (prev, dist)

    def all_paths(self, src, dst, max_hops=0):
        """Compute all paths to reach `dst` from `src` in the graph.

        Parameters
        ----------
        src: int
            The source node.
        dst : int
            The destination node.
        max_hops : int, optional
            The maximum number of hops to consider. 0 means infinite.

        Returns
        -------
        list
            A list of intermediate hops to reach `dst` from `src`.
        
        """

        paths = []
        self._all_paths(src, dst, max_hops, paths, dst, [])
        return paths

    def next_hop(self, src, dst):
        """Return the next hop on src to reach dst."""

        self._create_nexthop_matrix()
        return self._nexthop_matrix[dst][src]

    def distance(self, src, dst):
        """Return the minimum distance, in traversing cost, from src to dst."""

        self._create_nexthop_matrix()
        return self._distance_matrix[dst][src]

    def distance_path(self, src, dst, middle_nodes):
        """Return the distance to go from `src` to `dst` via `middle_nodes`."""

        distance = 0.
        last_node = src
        for curr_node in middle_nodes:
            distance += self.weight(last_node, curr_node)
            last_node = curr_node
        distance += self.weight(last_node, dst)
        return distance

    def diameter(self):
        """Return the maximum traversing cost between any two nodes."""

        self._create_nexthop_matrix()
        diameter = 0
        for _, neigh in self._distance_matrix.items():
            diameter = max(
                diameter,
                max([x for x in neigh.values() if x != float('inf')]))
        return diameter

    def farthest_nodes(self):
        """Return the set of pairs of nodes that are farthest from one another.
        """

        diameter = self.diameter() # also initializes self._distance_matrix
        ret = set()
        for u, neigh in self._distance_matrix.items():
            for v, dist in neigh.items():
                if dist == diameter:
                    ret.add((u, v))
        assert ret
        return ret

    def longest_path(self):
        """Return the length longest possible acyclic graph."""

        min_path = float('inf')
        for u in self._graph.keys():
            _, dist = self.spt(u, opposite_weight=True)
            min_path = min(min_path, min(dist.values()))

        return -min_path

    def save_dot(self, dotfilename):
        """Save the current network to a graph using Graphviz."""

        with open('{}.dot'.format(dotfilename), 'w') as dotfile:
            dotfile.write('digraph G {\n')
            dotfile.write('overlap=scale;\n')
            dotfile.write('node [shape=ellipse];\n')
            for u, neigh in self._graph.items():
                for v in neigh:
                    u_name = self.get_name_by_id(u)
                    v_name = self.get_name_by_id(v)
                    dotfile.write(f'{u_name} -> {v_name};\n')
                dotfile.write(f'{u_name} [shape=rectangle]\n')
            dotfile.write('}\n')

        subprocess.Popen(
            ['dot', '-Tpng',
             '-o{}.png'.format(dotfilename),
             '{}.dot'.format(dotfilename)])
        subprocess.Popen(
            ['dot', '-Tsvg',
             '-o{}.svg'.format(dotfilename),
             '{}.dot'.format(dotfilename)])

    def extract_bidirectional(self):
        """Return a new `Topology` object with only bidirectional edges.

        The edges that are not bidirectional are removed.

        As a result, the returned graph can have less nodes than the original.

        If the original graph has names assigned to nodes, they are re-used.

        Raises
        ------
        EmptyTopology
            The reduced graph is empty.

        """

        # Find bi-directional edges
        nodes_kept = set()
        edges = []
        for u, neigh in self._graph.items():
            for v in neigh:
                if v in self._graph and u in self._graph[v]:
                    edges.append([u, v])
                    edges.append([v, u])
                    nodes_kept.add(u)
                    nodes_kept.add(v)

        if not edges:
            raise EmptyTopology()
        
        # Create a new topology with the bi-directional edges only
        topo = Topology("edges", edges=edges)
        
        # Copy names
        node_names = dict()
        for node_id in nodes_kept:
            node_names[node_id] = self.get_name_by_id(node_id)
        topo.assign_names(node_names)

        # Copy weights
        topo._default_weight = self._default_weight
        for e in edges:
            weight = self.weight(e[1], e[0])
            if weight != topo._default_weight:
                topo.change_weight(e[1], e[0], weight)

        return topo

    def change_weight(self, src, dst, weight):
        """Change the weight betwen two nodes.

        Parameters
        ----------
        src : int
            The source node.
        dst : int
            The destination node.
        weight : float
            The weight or cost to reach `dst` from its neighbor `src`.

        Raises
        ------
        KeyError
            If `src` or `dst` do not exist or if they are not neighbors.
        
        """

        self._check_neighbors(src, dst)

        if weight == self._default_weight:
            return

        if self._weight_matrix is None:
            self._weight_matrix = dict()
        if dst not in self._weight_matrix:
            self._weight_matrix[dst] = dict()
        self._weight_matrix[dst][src] = weight

        self._invalidate()

    def change_all_weights(self, weight):
        """Reset all the weights to the same value.

        Parameters
        ----------
        weight : float
            The weight to be assigned to all edges.
        
        """

        self._default_weight = weight
        self._weight_matrix = dict()
        self._invalidate()

    def weight(self, src, dst):
        """Return the weight from `dst` to `src`.

        Parameters
        ----------
        src : int
            The source node.
        dst : int
            The destination node.            

        Returns
        -------
        The weight or cost to reach `dst` from its neighbor `src`.

        Raises
        ------
        KeyError
            If `src` or `dst` do not exist or if they are not neighbors.

        """

        self._check_neighbors(src, dst)
        if self._weight_matrix is not None:
            if dst in self._weight_matrix and src in self._weight_matrix[dst]:
                return self._weight_matrix[dst][src]
        return self._default_weight

    @staticmethod
    def traversing(spt, src, dst):
        """"Return the nodes traversed from src to dst according to the given SPT
        """

        ret = []
        nxt = src

        while nxt != dst:
            nxt = spt[nxt]
            if nxt != dst:
                ret.append(nxt)

        return ret

    def _all_paths(self, src, dst, max_hops, paths, cur_node, cur_path):
        for u in self._graph[cur_node]:
            if u == dst or u in cur_path:
                pass
            elif u == src:
                paths.append(cur_path)
            elif max_hops == 0 or len(cur_path) < max_hops:
                self._all_paths(src, dst, max_hops, paths, u, [u] + cur_path)

    def _check_neighbors(self, src, dst):
        if dst not in self._graph or src not in self._graph[dst]:
            raise KeyError(f"Nodes {dst} to {src} do not have an edge")

    def _create_incoming_id(self):
        # Do not overwrite a previously-created data structure
        if self._incoming_id is not None:
            return

        # Retrieve for every node the list of nodes with an incoming edge
        self._incoming_id = dict()
        incoming = dict()
        for u, neigh in self._graph.items():
            for v in neigh:
                if v not in incoming:
                    incoming[v] = []
                incoming[v].append(u)
        
        # Assign identifiers after sorting the list of incoming nodes
        for u, nodes in incoming.items():
            nodes.sort()
            id = 0
            for v in nodes:
                if u not in self._incoming_id:
                    self._incoming_id[u] = dict()
                self._incoming_id[u][v] = id
                id += 1

    def _create_nexthop_matrix(self):
        if self._nexthop_matrix is not None:
            assert self._distance_matrix is not None
            return
            
        self._nexthop_matrix = dict()
        self._distance_matrix = dict()

        for u in self._graph.keys():
            self._nexthop_matrix[u], self._distance_matrix[u] = self.spt(u)

    def _invalidate(self):
        """Invalidate the next-hop and distance matrices."""

        self._nexthop_matrix = None
        self._distance_matrix = None

class Topography:
    """Base class definining the interface of topography objects.

    A topography objects models a physical arrangement of nodes in space.
    
    """
    def __init__(self):
        pass

    def distance(self, src, dst):
        """Return the distance from `src` to `dst`.

        In general the distance from `dst` to `src` might be different.
        
        """
        raise NotImplementedError('This method must be overridden')

    def update_topology(self, topology):
        """Set the distance between nodes in `topology` according to this object.

        Parameters
        ----------
        topology: :class:`~.uiiit.topology.Topology`
            The topology where to set distances

        """
        for edge in topology.edges():
            topology.change_weight(edge[0], edge[1],
                                   self.distance(edge[0], edge[1]))

class TopographyDist(Topography):
    def __init__(self):
        super().__init__()
        self._distance = dict()

    def distance(self, src, dst):
        """Return the distance from `src` to `dst`, always symmetric."""

        if src == dst:
            return 0

        return self._distance[src][dst]

    def set_distance(self, u, v, dist):
        """Set the distance between nodes `u` and `v`.

        The distance between the two nodes is symmetric and it overwrites
        the previous value, if any.

        Parameters
        ----------
        u : int
            The first node identifier.
        v : int
            The second node identifier.
        dist : float
            The distance between the two nodes.
        
        """

        if u not in self._distance:
            self._distance[u] = dict()
        if v not in self._distance:
            self._distance[v] = dict()

        self._distance[u][v] = dist
        self._distance[v][u] = dist

    @staticmethod
    def make_from_topology(topology, distance_min, distance_max):
        """Return a `TopographyDist` object initialized based on the arguments.

        The distance between any two nodes that share an edge in `topology`
        is set to a random value drawn from a uniform r.v. in [`distance_min`,
        `distance_max`].

        Parameters
        ----------
        topology : `Topology`
            The topology from which to infer the edges. Only consider
            the bidirectional edges (unidirectional ones are ignored).
        distance_min : float
            The minimum distance between two nodes.
        distance_max : float
            The maximum distance between wo nodes.

        Returns
        -------
        A `TopographyDist` object.
        
        """

        if distance_min > distance_max:
            raise ValueError(f'Min distance ({distance_min}) cannot be greater than max distance ({distance_max})')

        topo = TopographyDist()
        for e in topology.biedges():
            dist =  random.uniform(distance_min, distance_max)
            topo.set_distance(e[0], e[1], dist)
        return topo

    def edges(self):
        """Return the list of (bidirectional) edges."""

        ret = []
        for u, neigh in self._distance.items():
            for v in neigh:
                ret.append([u, v])
        return ret
        
class Topography2D(TopographyDist):
    """Physical layout of nodes on a plane.

    The position is identified by means of a (x, y) coordinated in a plane.

    Parameters
    ----------
    type : { 'disc', 'square' }
        Type of layout.
    nodes : int, optional
        Number of nodes.
    size : float, optional
        Depends on the type: with a disc type this is the radius; with a
        square type this is edge size.
    threshold : float, optional
        Connect two nodes if their distance is below this threshold.

    Raises
    ------
    ValueError
        If an invalid type is provided by the user, or if the arguments
        are invalid.
    """
    def __init__(self, type, nodes=2, size=1, threshold=2):
        super().__init__()

        if nodes < 0:
            raise ValueError(f'The number of nodes cannot be negative: {nodes}')

        self._positions = dict()
        if type == 'disc':
            if size < 0:
                raise ValueError(f'The disc radius cannot be negative: {size}')
            for u in range(nodes):
                dist_from_origin = random.uniform(0, size)
                theta = random.uniform(0, math.pi)
                self._positions[u] = (
                    dist_from_origin * math.cos(theta),
                    dist_from_origin * math.sin(theta)
                )

        elif type == 'square':
            if size < 0:
                raise ValueError(f'The square edge size cannot be negative: {size}')
            for u in range(nodes):
                self._positions[u] = (
                    random.uniform(-size / 2, size / 2),
                    random.uniform(-size / 2, size / 2)
                )

        else:
            raise ValueError(f'Invalid Topography2D type: {type}')

        self._make_edges(threshold)

    def __getitem__(self, node):
        """Return the node's position."""

        return self._positions[node]

    def orphans(self):
        """"Return the list of disconnected nodes."""

        ret = set()
        for u in self._positions.keys():
            if u not in self._distance:
                ret.add(u)
        return ret

    def export(self, node_path, edge_path):
        """Export the topography to the given text files.
        
        Parameters
        ----------
        node_path : str
            The name of the file where to save the coordinates of nodes.
        edge_path : str
            The name of the file where to save the edges.

        """

        with open(node_path, 'w') as nodefp, open(edge_path, 'w') as edgefp:
            for u, pos in self._positions.items():
                nodefp.write(f'{u} {pos[0]}, {pos[1]}\n')
            for e in self.edges():
                u = e[0]
                v = e[1]
                u_pos = self._positions[u]
                v_pos = self._positions[v]
                edgefp.write(f'{u} {u_pos[0]} {u_pos[1]} {v} {v_pos[0]} {v_pos[1]}\n')

    def _make_edges(self, threshold):
        for u, u_pos in self._positions.items():
            for v, v_pos in self._positions.items():
                if u == v:
                    continue
                dist = math.sqrt(\
                    (u_pos[0] - v_pos[0]) ** 2 +\
                    (u_pos[1] - v_pos[1]) ** 2)
                if dist <= threshold:
                    self.set_distance(u, v, dist)
