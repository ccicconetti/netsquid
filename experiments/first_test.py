"""
Example experiment with quantum repeaters using uiiit modules.
"""

import random
import logging
import pandas

import pydynaa
import netsquid as ns
from netsquid.qubits import ketstates as ks
from netsquid.protocols import LocalProtocol, NodeProtocol, Protocol, Signals
from netsquid.components import Message, QuantumProgram
from netsquid.components.models.qerrormodels import FibreLossModel
from netsquid.components.instructions import INSTR_MEASURE_BELL, INSTR_X, INSTR_Z
from uiiit.qnetwork import QNetworkUniform
from uiiit.qrepeater import QRepeater
from uiiit.topology import Topology
from uiiit.traffic import SinglePairConstantApplication

__all__ = []

class SwapProtocol(NodeProtocol):
    class PathInfo:
        def __init__(self):
            self.x_corr = 0
            self.z_corr = 0
            self.counter = 0

        def incr(self, x_corr, z_corr):
            self.counter += 1
            self.x_corr += x_corr
            self.z_corr += z_corr

    class SwapProgram(QuantumProgram):
        """Quantum processor program that measures two qubits."""
        default_num_qubits = 2

        def program(self):
            q1, q2 = self.get_qubit_indices(2)
            self.apply(INSTR_MEASURE_BELL, [q1, q2], output_key="m", inplace=False)
            yield self.run()

    class CorrectProgram(QuantumProgram):
        """Quantum processor program that applies all swap corrections."""
        default_num_qubits = 1

        def set_corrections(self, x_corr, z_corr):
            self.x_corr = x_corr % 2
            self.z_corr = z_corr % 2

        def program(self):
            q1, = self.get_qubit_indices(1)
            if self.x_corr == 1:
                self.apply(INSTR_X, q1)
            if self.z_corr == 1:
                self.apply(INSTR_Z, q1)
            yield self.run()

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

        # Swap quantum program
        self._swap_program = SwapProtocol.SwapProgram()
        # self._swap_program = QuantumProgram(num_qubits=2)
        # q1, q2 = self._swap_program.get_qubit_indices(num_qubits=2)
        # self._swap_program.apply(INSTR_MEASURE_BELL, [q1, q2], output_key="m", inplace=False)

        # Correct quantum program
        self._correct_program = SwapProtocol.CorrectProgram()

        self._rx_messages = dict()

        # Discover classical channel ports
        self._cport_names = [x for x in self.node.ports]
        for port in self._cport_names:
            if 'ccon' not in port:
                self._cport_names.remove(port)

        qport_names = ' '.join([x for x in self._qmem.ports])
        cport_names = ' '.join([x for x in self._cport_names])
        logging.info(f"creating SwapProtocol on node {node.name}, qports: {qport_names}, cports: {cport_names}")

    def run(self):
        while True:
            # Create an event triggered when ALL the memory ports
            # receive a qubit: note that also lost qubits will trigger
            # such an event notification
            qevent = None
            for port in self._qmem.ports:
                if port in ["qin", "qout"]:
                    continue
                if qevent is None:
                    qevent = self.await_port_input(self._qmem.ports[port])
                else:
                    qevent &= self.await_port_input(self._qmem.ports[port])

            # Create an event triggered when ANY of the classical channel
            # connections receives a message
            cevent = None
            for port in self._cport_names:
                if cevent is None:
                    cevent = self.await_port_input(self.node.ports[port])
                else:
                    cevent |= self.await_port_input(self.node.ports[port])
                    
            # Wait until any of the two events happen
            expression = yield qevent | cevent

            if expression.first_term.value:
                # Qubits received on all the memory ports
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
                        pos1, pos2, dst_name, path = item
                        logging.debug(f"{ns.sim_time():.1f}: {self.node.name} ready to swap by measuring on {pos1} and {pos2}")
                        self.node.qmemory.execute_program(self._swap_program, qubit_mapping=[pos1, pos2])
                        yield self.await_program(self.node.qmemory)
                        m, = self._swap_program.output["m"]
                        m1, m2 = self._bsm_op_indices[m]

                        # Send result to one of the two parties creating the
                        # end-to-end entanglement
                        cchan = self._cport_name(dst_name)
                        logging.debug((f"{ns.sim_time():.1f}: {self.node.name} sending corrections [{m1}, {m2}] "
                                       f"(path {path}, timeslot {self._oracle.timeslot}) to {dst_name} via {cchan}"))
                        self.node.ports[cchan].tx_output(
                            Message([m1, m2], destination=dst_name,
                            path=path, timeslot=self._oracle.timeslot))
            else:
                # Message received from a classical channel
                for cport_name in self._cport_names:
                    port = self.node.ports[cport_name]
                    if not port.input_queue:
                        continue

                    msg = port.rx_input()
                    if msg.meta['destination'] == self.node.name:
                        # the message reached its final destination: correct qubit
                        if self._handle_correct_msg(msg):
                            yield self.await_program(self.node.qmemory, await_done=True, await_fail=True)
                            logging.debug(f"{ns.sim_time():.1f}: {self.node.name} corrections applied")
                            self._oracle.success(msg.meta['path'])
                            self.send_signal(Signals.SUCCESS)
                            self._rx_messages.clear()
                    else:
                        # we must forward the message to its next hop
                        self._forward(msg, dst_name)

    def _handle_correct_msg(self, msg):
        """Handle a new incoming correction message `msg`. Return if executed."""

        path = msg.meta['path']
        if path not in self._rx_messages:
            self._rx_messages[path] = SwapProtocol.PathInfo()

        path_info = self._rx_messages[path]
        path_length = self._oracle.path[path][4]
        mem_pos = self._oracle.path[path][3]

        m0, m1 = msg.items
        path_info.incr(m1, m0)
        
        if path_info.counter == path_length:
            if path_info.x_corr or path_info.z_corr:
                self._correct_program.set_corrections(path_info.x_corr, path_info.z_corr)
                logging.debug((f"{ns.sim_time():.1f}: {self.node.name} ready to apply corrections "
                               f"for path {path} to qubit {mem_pos} (status {str(self.node.qmemory.status)})"))
                self.node.qmemory.execute_program(self._correct_program, qubit_mapping=[mem_pos])
                return True
        
        return False

    def _forward(self, msg, dst_name):
        """Forward the given message `msg` to the node `dst_name`."""

        cchan = self._cport_name(dst_name)
        logging.debug((f"{ns.sim_time():.1f}: {self.node.name} forwarding message (path {msg.meta['path']}, "
                        f"timeslot {msg.meta['timeslot']}) to {dst_name} via {cchan}"))
        self.node.ports[cchan].tx_output(msg)

    def _cport_name(self, dst_name):
        """Return the port name to reach the given destination"""

        return f'ccon{self._oracle.channel_id(self.node.name, dst_name)}'         

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
        qubit teleported; in the fifth item the number of swaps along the path.
        This structure is overwritten at every new timeslot.

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
        self.path = dict()

        self.num_successful = 0

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
        self.path.clear()

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
            # there is a path between alice and bob
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
                self._topology.distance(alice, bob) - 1,
            ]
            # logging.debug(f"timeslot #{self.timeslot}, path {bob}, {', '.join([str(x) for x in swap_nodes])}, {alice}")

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

        self.num_successful += 1

        # Measure fidelity
        path = self.path[path_id]
        qubit_a, = self._network.nodes[path[0]].qmemory.peek([path[2]])
        qubit_b, = self._network.nodes[path[1]].qmemory.peek([path[3]])
        fidelity = ns.qubits.fidelity([qubit_a, qubit_b], ks.b00, squared=True)
        logging.debug((f"timeslot #{self.timeslot}, measured fidelity of e2e entanglement "
                       f"between {path[0]}:{path[2]} and {path[1]}:{path[3]}: {fidelity:.3f}"))

        

def run_simulation(num_nodes, node_distance, timeslots, seed):
    logging.info(f"starting simulation #{seed}: num_nodes = {num_nodes}, distance = {node_distance} km, messages = {timeslots}")
    ns.sim_reset()
    ns.set_random_state(seed=seed)
    random.seed(seed)
    est_runtime = (0.5 + 2 * num_nodes - 2) * node_distance * 5e3
    logging.debug(f"estimated maximum end-to-end delay = {est_runtime} ns")

    p_loss_init   = 0.1
    p_loss_length = 0.1
    network_factory = QNetworkUniform(
        node_distance=node_distance,
        node_distance_error=node_distance / 1000,
        source_frequency=1e9 / est_runtime,
        qerr_model=FibreLossModel(p_loss_init=p_loss_init, p_loss_length=p_loss_length))

    dephase_rate  = 1000 # Hz
    depol_rate    = 1000 # Hz
    gate_duration = 2    # ns
    qrepeater_factory = QRepeater(dephase_rate=dephase_rate,
                                  depol_rate=depol_rate,
                                  gate_duration=gate_duration)

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
    distance = 5 # km
    timeslots = 3
    seed = 42

    df = run_simulation(num_nodes=num_nodes,
                        node_distance=distance,
                        timeslots=timeslots,
                        seed=seed)

    logging.info(df)