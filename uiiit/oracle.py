"""This module specifies an Oracle used for routing with quantum repeaters.
"""

import logging

import netsquid as ns
from netsquid.qubits import ketstates as ks
from netsquid.protocols import Protocol, Signals

from uiiit.topology import Topology, EmptyTopology
from uiiit.utils import Chronometer

__all__ = [
    "Oracle"
    ]

class Oracle(Protocol):
    _minmax_dist_max_paths   = 2**24
    _minmax_dist_diam_factor = 2

    class Path:
        def __init__(self, alice_name, bob_name,
                     alice_edge_id, bob_edge_id,
                     swap_nodes, timestamp):
            self.alice_name    = alice_name
            self.bob_name      = bob_name
            self.alice_edge_id = alice_edge_id
            self.bob_edge_id   = bob_edge_id
            self.swap_nodes    = swap_nodes
            self.timestamp     = timestamp

        def __repr__(self):
            return (f'path between {self.alice_name} ({self.alice_edge_id}) '
                    f'and {self.bob_name} ({self.bob_edge_id}), with '
                    f'{len(self.swap_nodes)} swaps, '
                    f'established at time {self.timestamp:.1f}')

    """Network oracle: knows everything, can communicate at zero delay.

    Parameters
    ----------
    algorithm : { 'spf-hops', 'spf-dist', 'minmax-dist' }
        The algorithm to use for path selection.
    network : `netsquid.nodes.network.Network`
        The network of nodes.
    topology : `uiiit.topology.Topology`
        The `Topology` object.
    app : `uiiit.traffic.Application`
        The application that selects the nodes wishing to establish
        end-to-end entanglement timeslot by timeslot.
    stat : `uiiit.simstat.Stat`
        The statistics collection module.

    Properties
    ----------
    timeslot : int
        The current timeslot, counting from 0.
    mem_pos : dict
        A dictionary containing the outcome of the routing algorithm.
        For each element, the key is the name of the node that has to swap
        two qubits in its internal memory; the value is a list of tuples, where:
        the first two items hold the identifiers of the memory positions that
        have to be swapped; the third item is the name of the node to which
        the corrections have to be sent; the fourth item is an identifier of
        the qubit in this timeslot. This structure is overwritten at every new
        timeslot.
    path : dict
        A dictionary containing for each path (in the key) a tuple with the
        following data: in the first two items the names of the alice and bob
        nodes; in the third item the memory position on alice of the
        qubit teleported; in the fourth item the memory position on bob of the
        qubit teleported; in the fifth item the number of swaps along the path;
        the sixth item is the time when the entanglement started.
        This structure is overwritten at every new timeslot.

    """
    def __init__(self, algorithm, network, topology, app, stat):
        super().__init__(name="Oracle")

        self._algorithm = algorithm
        self._topology  = topology
        self._topology_hops = Topology('edges', edges=topology.edges())
        self._network   = network
        self._app       = app
        self._stat      = stat

        self._edges = []
        self._pending_nodes = set(topology.node_names)

        self.timeslot = 0
        self.mem_pos  = dict()
        self.path     = dict()

        self._all_paths = None
        if algorithm == 'minmax-dist':
            with Chronometer():
                self._initialize_min_max_dist()

        logging.debug(f"Create Oracle for network {network.name}, app {app.name}, nodes: {topology.node_names}")

    def link_good(self, node_name, positions):
        """Mark a link as good, i.e., entanglement has succeeded.

        Parameters
        ----------
        node_name : str
            The name of the node that has detected that entanglement has succeeded.
        positions : list
            The identifiers of the memory position where entanglement has been
            detected as successful.
        
        """

        # Check if this is a new timeslot
        if not self._edges:
            self.timeslot += 1

        # Add a new edge to the temporary graph for this timeslot
        for pos in positions:
            rx_node_id = self._topology.get_id_by_name(node_name)
            tx_node_id = self._topology.neigh_from_id(rx_node_id, pos)
            self._edges.append([rx_node_id, tx_node_id])

        # Remove the node from the pending list
        self._pending_nodes.remove(node_name)

        # Do nothing if some nodes did not mark their successes/failures yet
        if not self._pending_nodes:
            # All nodes have marked their successes/failures, time to do routing!
            self._routing()

    def _routing(self):
        """Perform routing based on the info received by the nodes."""

        # Remove previous entanglement data structure
        self.mem_pos.clear()
        self.path.clear()

        # Seek end-to-end entanglement paths with the current edges
        # The end-to-end pairs from the applications are served round-robin
        e2e_pairs = self._app.get_pairs(self.timeslot)
        path_id = 0
        cur_pair_ndx = len(e2e_pairs) # Beyond last element
        while e2e_pairs:
            # Wrap-around if the end is reached
            if cur_pair_ndx == len(e2e_pairs):
                cur_pair_ndx = 0 # Wrap-around
            cur_pair = e2e_pairs[cur_pair_ndx]
            if self._add_path(cur_pair[0], cur_pair[1], path_id) == False:
                # The current pair cannot be served in the reduced graph
                del e2e_pairs[cur_pair_ndx]
                continue
            if cur_pair[2] == 1:
                # All the entanglements have been served
                del e2e_pairs[cur_pair_ndx]
            else:
                if cur_pair[2] != 0: # Special value: means infinite
                    e2e_pairs[cur_pair_ndx] = (cur_pair[0], cur_pair[1], cur_pair[2]-1)
                cur_pair_ndx += 1
            path_id += 1
    
        logging.debug((f"{ns.sim_time():.1f}: timeslot #{self.timeslot}, "
                       f"found {path_id} end-to-end entanglement paths"))

        # Notify all nodes that they can proceed
        self.send_signal(Signals.SUCCESS)

        # Clear the edges with successful entanglement
        self._edges.clear()

        # Wait for all nodes again
        self._pending_nodes = set(self._topology.node_names)

    def _add_path(self, alice_name, bob_name, path_id):
        """Try to find a new end-to-end entanglement path.

        If a path is found, then `self.mem_pos` and `self.path` are
        added a new element, and the edges used are pruned from `self._edges`.

        Parameters
        ----------
        alice_name : str
            The name of the first node of the end-to-end entanglement path.
        bob_name : str
            The name of the first node of the end-to-end entanglement path.
        path_id : int
            The identifier of the path to use.
        
        Returns
        -------
            True if a path was found, False otherwise.

        """

        try:
            # Create a new graph with only the edges where entanglement has succeeded
            graph_uni = Topology("edges", edges=self._edges)
            graph_uni.copy_names(self._topology)
            graph_uni.copy_weights(self._topology)

            # Create a new reduced graph by removing unidirectional edges
            graph_bi = graph_uni.extract_bidirectional()

        except EmptyTopology:
            logging.debug((f"{ns.sim_time():.1f}: timeslot #{self.timeslot}, "
                           f"{alice_name} -> {bob_name} [path {path_id}]: "
                           f"empty reduced graph"))
            return False

        # logging.debug(f"{ns.sim_time():.1f}: timeslot #{self.timeslot}, graph {graph_uni}")
        logging.debug(f"{ns.sim_time():.1f}: timeslot #{self.timeslot}, reduced graph {graph_bi}")
        # graph_bi.save_dot(f"graph_bi{self.timeslot}")

        # Retrieve from the application the list of pairs with e2e entanglement
        alice = graph_bi.get_id_by_name(alice_name) if alice_name in graph_bi.node_names else None
        bob = graph_bi.get_id_by_name(bob_name) if bob_name in graph_bi.node_names else None

        # Search the path from bob to alice, but only if both are still in
        # the reduced graph
        swap_nodes = self._path_selection(bob, alice, graph_bi)

        if swap_nodes is None:
            logging.debug((f"{ns.sim_time():.1f}: timeslot #{self.timeslot}, "
                           f"{alice_name} -> {bob_name} [path {path_id}]: "
                           f"no way to create an e2e entanglement path"))
            return False

        # There is a path between alice and bob
        assert alice is not None
        assert bob is not None
        assert swap_nodes is not None

        alice_nxt = swap_nodes[-1] if swap_nodes else bob
        bob_prv = swap_nodes[0] if swap_nodes else alice
        self.path[path_id] = Oracle.Path(
            alice_name,
            bob_name,
            self._topology.incoming_id(alice, alice_nxt),
            self._topology.incoming_id(bob, bob_prv),
            swap_nodes,
            ns.sim_time()
        )

        # If there are no intermediate nodes, then alice and bob shared
        # an entangling connection, hence there is no need to send
        # out corrections, and the qubits received can be used immediately
        if not swap_nodes:
            self._edges.remove([alice, bob])
            self._edges.remove([bob, alice])
            self.success(path_id)
            return True

        for i in range(len(swap_nodes)):
            cur = swap_nodes[i]
            prv = bob if i == 0 else swap_nodes[i-1]
            nxt = alice if i == (len(swap_nodes)-1) else swap_nodes[i+1]
            prv_pos = self._topology.incoming_id(cur, prv)
            nxt_pos = self._topology.incoming_id(cur, nxt)
            logging.debug(
                (f"{ns.sim_time():.1f}: timeslot #{self.timeslot}, "
                 f"{alice_name} -> {bob_name} [path {path_id}]: "
                 f"on node {cur} entangle node {prv} (mem pos {prv_pos}) "
                 f"and node {nxt} (mem pos {nxt_pos})"))

            self._edges.remove([cur, prv])
            self._edges.remove([prv, cur])

            cur_name = self._topology.get_name_by_id(cur)
            if cur_name not in self.mem_pos:
                self.mem_pos[cur_name] = []
            self.mem_pos[cur_name].append([
                prv_pos, nxt_pos,
                self._topology.get_name_by_id(bob),
                path_id])

        self._edges.remove([alice, alice_nxt])
        self._edges.remove([alice_nxt, alice])                

        return True

    def _path_selection(self, src, dst, graph_bi):
        """Return the list of intermediate nodes to perform swapping.
        
        Parameters
        ----------
        src : int
            The source node
        dst : int
            The destination node
        graph_bi : `Topology`
            The reduced bi-directional graph. May be changed by the method.

        """

        if src is None or dst is None:
            return None
        
        # The nodes src and dst may be None, but if they aren't then
        # we assume they both exist in the bidirectional graph
        nodes = graph_bi.nodes()
        assert src in nodes
        assert dst in nodes

        if self._algorithm in ['spf-hops', 'spf-dist']:
            if self._algorithm == 'spf-hops':
                graph_bi.change_all_weights(1)
            prev, _ = graph_bi.spt(dst)
            
            if prev is None or prev[src] is None:
                return None
            return Topology.traversing(prev, src, dst)

        elif self._algorithm == 'minmax-dist':
            assert self._all_paths is not None

            curr = None
            for path in self._all_paths[src][dst]:
                # Discard path if cannot be implemented in the reduced graph.
                not_usable = False
                full_path = [src] + path[0] + [dst]
                for i in range(len(full_path)-1):
                    if full_path[i+1] not in nodes or \
                        graph_bi.isedge(full_path[i], full_path[i+1]) == False:
                        not_usable = True
                        break
                if not_usable:
                    continue
                if curr is None or curr[1] > path[1]:
                    curr = path

            return curr[0] if curr is not None else None

        raise NotImplementedError(
            f'Unknown path selection algorithm: {self._algorithm}')

    def channel_id(self, src, dst):
        """Return the channel identifier where to send a message.

        Parameters
        ----------
        src : str
            Name of the current node that will send the message.
        dst : str
            Name of the destination node.
        
        Returns
        -------
        int
            The identifier of the channel where to send the message.
        
        """

        src_id = self._topology.get_id_by_name(src)
        dst_id = self._topology.get_id_by_name(dst)
        nxt_id = self._topology.next_hop(src_id, dst_id)
        return self._topology.incoming_id(src_id, nxt_id)
        
    def success(self, path_id):
        """The path `path_id` in this timeslot is successful."""

        path = self.path[path_id]

        # Distance on the original topology of the two end nodes
        dist = len(path.swap_nodes)

        # Measure fidelity
        qubit_a, = self._network.nodes[path.alice_name].qmemory.peek([path.alice_edge_id])
        qubit_b, = self._network.nodes[path.bob_name].qmemory.peek([path.bob_edge_id])
        fidelity = ns.qubits.fidelity([qubit_a, qubit_b], ks.b00, squared=True)
        self._stat.add(f"fidelity-{dist}", fidelity)

        # Record latency as the time between when the entanglement was ready
        # at each node and the time when all the corrections have been applied
        # to one of the end nodes.
        latency = ns.sim_time() - path.timestamp
        self._stat.add(f"latency-{dist}", latency * 1e-6) # convert ns to ms

        # Record the physical distance, in the shorted path, of e2e entanglement.
        self._stat.add(f"distance-{dist}",
                       self._topology.distance_path(
                           self._topology.get_id_by_name(path.bob_name),
                           self._topology.get_id_by_name(path.alice_name),
                           path.swap_nodes))

        # Counter of successful e2e entanglements.
        fid_thresholds = [0, 0.6, 0.7, 0.8, 0.9]
        for fid in fid_thresholds:
            if fidelity > fid:
                self._stat.count(f"success-{fid:.1f}", 1)

        logging.debug((f"{ns.sim_time():.1f}: "
                       f"timeslot #{self.timeslot}, e2e entanglement path {path_id} "
                       f"{path}, distance {dist}: "
                       f"fidelity {fidelity:.3f}, latency {latency:.3f}"))
        
        # Remove the path once it is found to be successful
        del self.path[path_id]

    def _initialize_min_max_dist(self):
        diameter = Topology("edges", edges=self._topology.edges()).diameter()
        nodes = self._topology.nodes()
        self._all_paths = dict()
        counter = 0
        for u in nodes:
            self._all_paths[u] = dict()
            for v in nodes:
                if u == v:
                    continue
                self._all_paths[u][v] = []
                curr = self._all_paths[u][v]
                for p in self._topology.all_paths(
                    u, v, self._minmax_dist_diam_factor * diameter):
                    max_cost = 0
                    for swap_node in p:
                        max_cost = max(max_cost,
                                       self._topology.distance(swap_node, v))
                    curr.append((p, max_cost))
                    counter += 1
                    if counter == self._minmax_dist_max_paths:
                        raise ValueError('Too many paths, bailing out')

        if logging.getLogger().isEnabledFor(logging.DEBUG):
            for src, destinations in self._all_paths.items():
                for dst, paths in destinations.items():
                    logging.debug(f'{src} -> {dst}')
                    for path, cost in paths:
                        logging.debug(f'path {path} cost {cost}')