"""
Example experiment with quantum repeaters using uiiit modules.
"""

import random
import logging
import pandas

import pydynaa
import netsquid as ns
from netsquid.protocols import LocalProtocol, NodeProtocol, Protocol, Signals
from netsquid.components import Message, QuantumProgram
from netsquid.components.models.qerrormodels import FibreLossModel
from netsquid.components.instructions import INSTR_MEASURE_BELL
from uiiit.qnetwork import QNetworkUniform
from uiiit.qrepeater import QRepeater
from uiiit.topology import Topology
from uiiit.traffic import SinglePairConstantApplication

__all__ = []

class SwapProtocol(NodeProtocol):
    """Perform Swap on a repeater node.

    Parameters
    ----------
    name : str
        Name of this protocol.
    node : `netsquid.nodes.node.Node`
        Node this protocol runs on.
    oracle : `Oracle`
        The oracle.

    """
    _bsm_op_indices = [(0, 0), (0, 1), (1, 0), (1, 1)]

    def __init__(self, name, node, oracle):
        super().__init__(node, name)
        self._oracle = oracle
        self._qmem = self.node.qmemory
        self._program = QuantumProgram(num_qubits=2)
        q1, q2 = self._program.get_qubit_indices(num_qubits=2)
        self._program.apply(INSTR_MEASURE_BELL, [q1, q2], output_key="m", inplace=False)

        port_names = ' '.join([x for x in self._qmem.ports])
        logging.info(f"creating SwapProtocol on node {node.name}, ports: {port_names}")

    def run(self):
        while True:
            # Wait for the qubits to arrive
            event = None
            for port in self._qmem.ports:
                if port in ["qin", "qout"]:
                    continue
                if event is None:
                    event = self.await_port_input(self._qmem.ports[port])
                else:
                    event &= self.await_port_input(self._qmem.ports[port])
            yield event

            logging.debug((f"{ns.sim_time():.1f}: {self._qmem.name} "
                           f"{self._qmem.num_used_positions}/{self._qmem.num_positions} "
                           f"received (empty: {self._qmem.unused_positions})"))

            positions = []
            for pos in range(self._qmem.num_positions):
                if pos not in self._qmem.unused_positions:
                    positions.append(pos)
            self._oracle.link_good(self.node.name, positions)

            # Wait for the oracle to take its decisions
            yield self.await_signal(self._oracle, Signals.SUCCESS)

            # Entangle the memory positions as specified by the oracle
            if self.node.name in self._oracle.mem_pos:
                for item in self._oracle.mem_pos[self.node.name]:
                    pos1, pos2, dst_name = item
                    logging.debug(f"{ns.sim_time():.1f}: {self.node.name} ready to swap by measuring on {pos1} and {pos2}")
                    self.node.qmemory.execute_program(self._program, qubit_mapping=[pos1, pos2])
                    yield self.await_program(self.node.qmemory)
                    m, = self._program.output["m"]
                    m1, m2 = self._bsm_op_indices[m]

                    # Send result to one of the two parties creating the
                    # end-to-end entanglement
                    cchan = f'ccon{self._oracle.channel_id(self.node.name, dst_name)}'
                    logging.debug(f"{ns.sim_time():.1f}: {self.node.name} sending corrections ({m1}, {m2}) to {dst_name} via {cchan}")
                    # self.node.ports["ccon_R"].tx_output(Message([m1, m2], path=0, timeslot=self.oracle.timeslot))

            # Pop all non-empty but unused memory positions
            # XXX

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

    Properties
    ----------
    timeslot : int
        The current timeslot, counting from 0.
    mem_pos : dist
        A dictionary containing the outcome of the routing algorithm.
        For each element, the key is the name of the node that has to swap
        two qubits in its internal memory; the value is a list of tuples, where
        the first two items hold the identifiers of the memory positions that
        have to be swapped and the third item is the name of the node to which
        the corrections have to be sent.

    """
    def __init__(self, network, topology, app):
        super().__init__(name="Oracle")

        self._topology = topology
        self._network = network
        self._app = app

        self._edges = []
        self._pending_nodes = set(topology.node_names)

        self.timeslot = 0
        self.mem_pos = dict()

        print(f"Create Oracle for network {network.name}, app {app.name}, nodes: {topology.node_names}")

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

        # If all the nodes have marked their successes/failures, then move ahead
        if self._pending_nodes:
            return

        # remove previous entanglement data structure
        self.mem_pos.clear()

        # Create a new graph with only the edges where entanglement has succeeded
        graph_uni = Topology("edges", edges=self._edges)
        graph_uni.copy_names(self._topology)
        logging.debug(f"timeslot #{self.timeslot}, graph {graph_uni}")
        graph_bi = graph_uni.extract_bidirectional()
        logging.debug(f"timeslot #{self.timeslot}, reduced graph {graph_bi}")
        # graph_bi.save_dot(f"graph_bi{self.timeslot}")

        # Retrieve from the application the list of pairs with e2e entanglement
        pairs = self._app.get_pairs(self.timeslot)

        assert len(pairs) == 1
        alice = graph_bi.get_id_by_name(pairs[0][0])
        bob = graph_bi.get_id_by_name(pairs[0][1])
        prev, _ = graph_bi.spt(alice)
        if prev[bob] is None:
            logging.debug(f"timeslot #{self.timeslot}, no way to create e2e entanglement between {pairs[0][0]} and {pairs[0][1]}")

        else:
            # there is a path between alice and bob
            swap_nodes = Topology.traversing(prev, bob, alice)
            # logging.debug(f"timeslot #{self.timeslot}, path {bob}, {', '.join([str(x) for x in swap_nodes])}, {alice}")

            for i in range(len(swap_nodes)):
                cur = swap_nodes[i]
                prv = bob if i == 0 else swap_nodes[i-1]
                nxt = alice if i == (len(swap_nodes)-1) else swap_nodes[i+1]
                prv_pos = self._topology.incoming_id(cur, prv)
                nxt_pos = self._topology.incoming_id(cur, nxt)
                logging.debug((f"timeslot #{self.timeslot}, e2e entanglement between {pairs[0][0]} and {pairs[0][1]}: "
                               f"on node {cur} entangle node {prv} (mem pos {prv_pos}) and node {nxt} (mem pos {nxt_pos})"))

                cur_name = self._topology.get_name_by_id(cur)
                if cur not in self.mem_pos:
                    self.mem_pos[cur_name] = []
                self.mem_pos[cur_name].append([
                    prv_pos, nxt_pos,
                    self._topology.get_name_by_id(bob)])

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
        
    def set_src_node(self, node):
        logging.debug(f"{ns.sim_time():.1f}: {self.name} {node.name} is the SRC node")
        self.src_node = node.name

    def set_dst_node(self, node):
        logging.debug(f"{ns.sim_time():.1f}: {self.name} {node.name} is the DST node")
        self.dst_node = node.name

    def success(self):
        self.num_successful += 1

    def path_length(self, path):
        if path != 0:
            raise Exception(f"Unknown path {path}")
        return len(self.node_names) - 2

def run_simulation(num_nodes, node_distance, timeslots, seed):
    logging.info(f"starting simulation #{seed}: num_nodes = {num_nodes}, distance = {node_distance} km, messages = {timeslots}")
    ns.sim_reset()
    ns.set_random_state(seed=seed)
    random.seed(seed)
    est_runtime = (0.5 + 2 * num_nodes - 2) * node_distance * 5e3
    logging.debug(f"estimated maximum end-to-end delay = {est_runtime} ns")

    p_loss_init = 0.1
    p_loss_length = 0.1
    network_factory = QNetworkUniform(
        node_distance=node_distance,
        node_distance_error=node_distance / 1000,
        source_frequency=1e9 / est_runtime,
        qerr_model=FibreLossModel(p_loss_init=p_loss_init, p_loss_length=p_loss_length))

    qrepeater_factory = QRepeater(1e6, 1e6, 1)

    topology = Topology("grid", size=num_nodes)

    network = network_factory.make_network(
        name="QNetworkUniform",
        qrepeater_factory=qrepeater_factory,
        topology=topology)

    # List of nodes, sorted in lexycographic order by their names
    node_names = sorted(network.nodes.keys())
    node_names_dict = dict()
    for i in range(len(node_names)):
        node_names_dict[i] = node_names[i]
    topology.assign_names(node_names_dict)
    nodes = [network.nodes[name] for name in node_names]

    protocol = LocalProtocol(nodes=network.nodes)

    # Create the application that will select the pairs of nodes wishing
    # to share entanglement
    app = SinglePairConstantApplication("ConstApp", "Node_0", "Node_5")

    # Create the oracle and add it to the local protocol
    oracle = Oracle(network, topology, app)
    protocol.add_subprotocol(oracle)

    # Add SwapProtocol to all repeater nodes. Note: we use unique names,
    # since the subprotocols would otherwise overwrite each other in the main protocol.
    for node in nodes:
        subprotocol = SwapProtocol(name=f"Swap_{node.name}", node=node, oracle=oracle)
        protocol.add_subprotocol(subprotocol)
    protocol.start()

    # Start simulation
    ns.sim_run(est_runtime * timeslots)

    df = pandas.DataFrame()

    return df

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    ns.set_qstate_formalism(ns.QFormalism.DM)

    num_nodes=3
    distance = 6.25 # km
    timeslots = 2
    seed = 42

    df = run_simulation(num_nodes=num_nodes,
                        node_distance=distance,
                        timeslots=timeslots,
                        seed=seed)

    logging.info(df)