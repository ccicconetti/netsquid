"""This module specifies a class that models the topology of a network.
"""

import subprocess

__all__ = [
    "Topology"
    ]

class EmptyTopology(Exception):
    def __init__(self):
        self.message = "Empty graph"

class Topology:
    """Topology of a network of nodes.
    
    Parameters
    ----------
    type : { 'chain', 'grid', 'brite', 'edges' }
        Type of the network to create.
    size : int, optional
        The actual meaning depends on the network type: 
        with "grid", this is the number of nodes on any side of the square;
        with "chain", it's the size of the chain.
        It is unused with other topology types.
    in_file_name : str, optional
        Name of the input file. Only used with a "brite" topology.
    edges : list, optional
        List of edges [dst -> src]. Only used with an "edges" topology.

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
    def __init__(self, type, size=1, in_file_name="", edges=[]):
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

        # Create the graph
        if type == "chain":
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
        """Return the list of pairs of node which have a bidirectional connection
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

    def spt(self, source):
        """Compute the shortest-path tree to source
        """

        Q = []
        dist = {}
        prev = {}
        for v in self._graph.keys():
            Q.append(v)
            dist[v] = 100000
            prev[v] = None

        if source not in dist:
            raise RuntimeError(f'Cannot compute the path to node {source} that is not in {Q}')

        dist[source] = 0

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
                alt = dist[u] + 1
                if alt < dist[v]:
                    dist[v] = alt
                    prev[v] = u

        return (prev, dist)

    def next_hop(self, src, dst):
        """Return the next hop on src to reach dst."""

        self._create_nexthop_matrix()
        return self._nexthop_matrix[dst][src]

    def distance(self, src, dst):
        """Return the distance, in hops, from src to dst."""

        self._create_nexthop_matrix()
        return self._distance_matrix[dst][src]

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
        
        topo = Topology("edges", edges=edges)
        
        node_names = dict()
        for node_id in nodes_kept:
            node_names[node_id] = self.get_name_by_id(node_id)

        topo.assign_names(node_names)

        return topo

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
    