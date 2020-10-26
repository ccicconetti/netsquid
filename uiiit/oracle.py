"""This module specifies an Oracle used for routing with quantum repeaters.
"""

import logging

import netsquid as ns
from netsquid.qubits import ketstates as ks
from netsquid.protocols import Protocol, Signals

from uiiit.topology import Topology

__all__ = [
    "Oracle"
    ]

class Oracle(Protocol):    
    """Network oracle: knows everything, can communicate at zero delay.

    Parameters
    ----------
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
    def __init__(self, network, topology, app, stat):
        super().__init__(name="Oracle")

        self._topology = topology
        self._network  = network
        self._app      = app
        self._stat     = stat

        self._edges = []
        self._pending_nodes = set(topology.node_names)

        self.timeslot = 0
        self.mem_pos  = dict()
        self.path     = dict()

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

        # Add a new edge to the temporary graph for this timeslot
        for pos in positions:
            rx_node_id = self._topology.get_id_by_name(node_name)
            tx_node_id = self._topology.neigh_from_id(rx_node_id, pos)
            self._edges.append([rx_node_id, tx_node_id])

        # Remove the node from the pending list
        self._pending_nodes.remove(node_name)

        # Do nothing if some nodes did not mark their successes/failures yet
        if self._pending_nodes:
            return

        #
        # All nodes have marked their successes/failures, time to do routing!
        #

        # Remove previous entanglement data structure
        self.mem_pos.clear()
        self.path.clear()

        try:
            # Create a new graph with only the edges where entanglement has succeeded
            graph_uni = Topology("edges", edges=self._edges)
            graph_uni.copy_names(self._topology)

            # Create a new reduced graph by removing unidirectional edges 
            graph_bi = graph_uni.extract_bidirectional()

            logging.debug(f"timeslot #{self.timeslot}, graph {graph_uni}")
            logging.debug(f"timeslot #{self.timeslot}, reduced graph {graph_bi}")
            # graph_bi.save_dot(f"graph_bi{self.timeslot}")

            # Retrieve from the application the list of pairs with e2e entanglement
            pairs = self._app.get_pairs(self.timeslot)
            assert len(pairs) == 1
            alice_name = pairs[0][0]
            bob_name = pairs[0][1]
            alice = graph_bi.get_id_by_name(alice_name) if alice_name in graph_bi.node_names else None
            bob = graph_bi.get_id_by_name(bob_name) if bob_name in graph_bi.node_names else None

            # Search the path from bob to alice, but only if both are still in
            # the reduced graph
            prev = None
            if alice is not None and bob is not None:
                prev, _ = graph_bi.spt(alice)
            
            if prev is None or prev[bob] is None:
                logging.debug(f"timeslot #{self.timeslot}, no way to create e2e entanglement between {alice_name} and {bob_name}")

            else:
                # There is a path between alice and bob
                assert alice is not None
                assert bob is not None

                swap_nodes = Topology.traversing(prev, bob, alice)
                alice_nxt = swap_nodes[-1] if swap_nodes else bob
                bob_prv = swap_nodes[0] if swap_nodes else alice
                self.path[0] = [
                    alice_name,
                    bob_name,
                    self._topology.incoming_id(alice, alice_nxt),
                    self._topology.incoming_id(bob, bob_prv),
                    graph_bi.distance(alice, bob) - 1,
                    ns.sim_time(),
                ]
                # logging.debug(f"timeslot #{self.timeslot}, path {bob}, {', '.join([str(x) for x in swap_nodes])}, {alice}")

                # If there are no intermediate nodes, then alice and bob shared
                # an entangling connection, hence there is no need to send
                # out corrections, and the qubits received can be used immediately
                if not swap_nodes:
                    self.success(0)

                for i in range(len(swap_nodes)):
                    cur = swap_nodes[i]
                    prv = bob if i == 0 else swap_nodes[i-1]
                    nxt = alice if i == (len(swap_nodes)-1) else swap_nodes[i+1]
                    prv_pos = self._topology.incoming_id(cur, prv)
                    nxt_pos = self._topology.incoming_id(cur, nxt)
                    logging.debug((f"timeslot #{self.timeslot}, e2e entanglement between {alice_name} and {bob_name}: "
                                f"on node {cur} entangle node {prv} (mem pos {prv_pos}) and node {nxt} (mem pos {nxt_pos})"))

                    cur_name = self._topology.get_name_by_id(cur)
                    if cur not in self.mem_pos:
                        self.mem_pos[cur_name] = []
                    self.mem_pos[cur_name].append([
                        prv_pos, nxt_pos,
                        self._topology.get_name_by_id(bob),
                        0])
        except:
            logging.debug(f"timeslot #{self.timeslot}, reduced graph is empty")

        # Notify all nodes that they can proceed
        self.send_signal(Signals.SUCCESS)

        # This is a new timeslot
        self.timeslot += 1

        # Clear the edges with successful entanglement
        self._edges.clear()

        # Wait for all nodes again
        self._pending_nodes = set(self._topology.node_names)

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
        dist = self._topology.distance(
            self._topology.get_id_by_name(path[0]),
            self._topology.get_id_by_name(path[1]))

        # Measure fidelity
        qubit_a, = self._network.nodes[path[0]].qmemory.peek([path[2]])
        qubit_b, = self._network.nodes[path[1]].qmemory.peek([path[3]])
        fidelity = ns.qubits.fidelity([qubit_a, qubit_b], ks.b00, squared=True)
        self._stat.add(f"fidelity-{dist}", fidelity)

        # Record latency as the time between when the entanglement was ready
        # at each node and the time when all the corrections have been applied
        # to one of the end nodes.
        latency = ns.sim_time() - path[5]
        self._stat.add(f"latency-{dist}", latency)

        # Record the number of swap required to realise e2e entanglement.
        self._stat.add(f"swap-{dist}", path[4])

        # Counter of successful e2e entanglements.
        self._stat.count("success", 1)
        if fidelity > 0.75:
            self._stat.count("success-0.75", 1)

        logging.debug((f"timeslot #{self.timeslot}, e2e entanglement between "
                f"{path[0]}:{path[2]} and {path[1]}:{path[3]} (distance {dist}): "
                f"fidelity {fidelity:.3f}, latency {latency:.3f} "
                f"swaps {path[4]}"))