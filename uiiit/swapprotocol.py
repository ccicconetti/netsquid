"""This module specifies the protocol used by the quantum repeaters.
"""

import logging

import netsquid as ns
from netsquid.protocols import NodeProtocol, Protocol, Signals
from netsquid.components import Message, QuantumProgram
from netsquid.components.instructions import INSTR_MEASURE_BELL, INSTR_X, INSTR_Z

__all__ = [
    "SwapProtocol"
    ]

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
        logging.debug(f"creating SwapProtocol on node {node.name}, qports: {qport_names}, cports: {cport_names}")

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
                        msg =  Message([m1, m2], source=self.node.name, destination=dst_name,
                                       path=path, timeslot=self._oracle.timeslot)
                        self.node.ports[cchan].tx_output(msg)
            else:
                # Message received from a classical channel
                for cport_name in self._cport_names:
                    port = self.node.ports[cport_name]
                    if not port.input_queue:
                        continue

                    msg = port.rx_input()
                    if msg.meta['destination'] != self.node.name:
                        # We must forward the message to its next hop
                        self._forward(msg)
                        continue

                    #
                    # The message reached its final destination: correct qubit
                    #

                    # From NetSquid documentation:
                    #
                    # https://docs.netsquid.org/latest-release/api_components/netsquid.components.channel.html
                    #
                    # If multiple items are send onto a channel during
                    # the same time instance, then all items become part
                    # of the same message with the same arrival time.
                    # When getting a message with multiple items, all
                    # items can be retrieved at once, or individual
                    # items can be got by their indices.
                    #
                    # Thus, we might have to unpack the items at pairs.
                    #
                    # NOTE that this way the metadata info has been lost.
                    #
                    program_executed = None
                    for i in range(0, len(msg.items), 2):
                        m0, m1 = msg.items[i:i+2]
                        all_received, program_executed = self._handle_correct_msg(msg.meta['path'], m0, m1)

                    assert program_executed is not None

                    # Execute the quantum program only if there are
                    # corrections to apply.
                    if program_executed:
                        yield self.await_program(self.node.qmemory, await_done=True, await_fail=True)

                    # In any case, if all the messages expected to be received
                    # have been received, we can notify the Oracle that the
                    # e2e entanglement is complete.
                    if all_received:
                        logging.debug(f"{ns.sim_time():.1f}: {self.node.name} corrections applied")
                        self._oracle.success(msg.meta['path'])
                        self.send_signal(Signals.SUCCESS)
                        del self._rx_messages[msg.meta['path']]

    def _handle_correct_msg(self, path, m0, m1):
        """Handle a new incoming correction message `msg`.
        
        Parameters
        ----------
        path
            The identifier of the path in this timeslot.
        m0
            The first correction factor (Z gate).
        m1
            The second correction factor (X gate).
        
        Returns
        -------
        (bool, bool)
            The first item of the pair is True if all the corrections have
            been received. The second item is True if a quantum program has
            been put into execution (and it has to be waited for). 

        """

        if path not in self._rx_messages:
            self._rx_messages[path] = SwapProtocol.PathInfo()

        path_info = self._rx_messages[path]
        path_length = self._oracle.path[path][4]
        mem_pos = self._oracle.path[path][3]

        path_info.incr(m1, m0)
        
        if path_info.counter < path_length:
            return (False, False)

        assert path_info.counter == path_length
        if path_info.x_corr or path_info.z_corr:
            self._correct_program.set_corrections(path_info.x_corr, path_info.z_corr)
            logging.debug((f"{ns.sim_time():.1f}: {self.node.name} ready to apply corrections "
                            f"for path {path} to qubit {mem_pos} (status {str(self.node.qmemory.status)})"))
            self.node.qmemory.execute_program(self._correct_program, qubit_mapping=[mem_pos])
            return (True, True)
        return (True, False)

    def _forward(self, msg):
        """Forward the given message `msg` to its destination node."""

        dst_name = msg.meta['destination']
        cchan = self._cport_name(dst_name)
        logging.debug((f"{ns.sim_time():.1f}: {self.node.name} forwarding message (path {msg.meta['path']}, "
                        f"timeslot {msg.meta['timeslot']}) to {dst_name} via {cchan}"))
        self.node.ports[cchan].tx_output(msg)

    def _cport_name(self, dst_name):
        """Return the port name to reach the given destination"""

        return f'ccon{self._oracle.channel_id(self.node.name, dst_name)}'         
