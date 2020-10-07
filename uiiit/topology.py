"""This module specifies a class that models the topology of a network.
"""

import subprocess

__all__ = [
    "Topology"
    ]

class Topology:
    """Topology of a network of nodes.
    
    Parameters
    ----------
    type : str
        Type of the network to create, one of: chain, grid, brite.
    size : int, optional
        The actual meaning depends on the network type: 
        with grid, this is the number of nodes on any side of the square;
        with chain, it's the size of the chain.
        It is unused with other topology types.
    in_file_name : str, optional
        Name of the input file. Only used with a brite topology.

    Properties
    ----------
    num_nodes : int
        Number of nodes in the network.

    Raises
    ------
    ValueError
        If the type is not supported.
    """
    def __init__(self, type, size=1, in_file_name=""):
        # The graph stored as a dictionary where:
        # key: node's name
        # value: all the nodes that have an edge with the node in the key as a set 
        self._graph = dict()

        if type == "chain":
            if size < 1:
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
            if size < 1:
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
            print(in_file_name)
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

                self.num_nodes = len(self._graph)

        else:
            raise ValueError(f'Invalid topology type: {type}')

    def __repr__(self):
        return str(self._graph)

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

        while len(Q) > 0:
            u = None
            last_value = None
            for node in Q:
                value = dist[node]
                if not u or value < last_value:
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

    def save_dot(self, dotfilename):
        "Save the current network to a graph using Graphviz"

        with open('{}.dot'.format(dotfilename), 'w') as dotfile:
            dotfile.write('graph G {\n')
            dotfile.write('overlap=scale;\n')
            dotfile.write('node [shape=ellipse];\n')
            edges = []
            for u, neigh in self._graph.items():
                for v in neigh:
                    new_edge = [u,v]
                    new_edge.sort()
                    if new_edge in edges:
                        continue
                    edges.append(new_edge)
                    dotfile.write(f'{u} -- {v};\n')
                dotfile.write(f'{u} [shape=rectangle]\n')
            dotfile.write('}\n')

        subprocess.Popen(
            ['neato', '-Tpng',
             '-o{}.png'.format(dotfilename),
             '{}.dot'.format(dotfilename)])
        subprocess.Popen(
            ['neato', '-Tsvg',
             '-o{}.svg'.format(dotfilename),
             '{}.dot'.format(dotfilename)])

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

