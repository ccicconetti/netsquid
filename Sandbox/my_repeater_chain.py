"""

In this example we show how a simple quantum repeater chain network can be setup and simulated in NetSquid.
The module file used in this example can be located as follows:

>>> import netsquid as ns
>>> logging.debug("This example module is located at: {}".format(
...       ns.examples.repeater_chain.__file__))
This example module is located at: .../netsquid/examples/repeater_chain.py

In the `repeater example <netsquid.examples.repeater.html>`_ we simulated a single quantum repeater on a network topology consisting of two nodes
connected via a single repeater node (see for instance `[Briegel et al.] <https://arxiv.org/abs/quant-ph/9803056>`_ for more background).
To simulate a repeater chain we will extend this network topology to be a line of *N* nodes as shown below:

.. aafig::
    :textual:
    :proportional:

    +-----------+   +----------+         +------------+   +----------+
    |           |   |          |         |            |   |          |
    | "Node 0"  O---O "Node 1" O-- ooo --O "Node N-1" O---O "Node N" |
    | "(Alice)" |   |          |         |            |   | "(Bob)"  |
    |           |   |          |         |            |   |          |
    +-----------+   +----------+         +------------+   +----------+

We will refer to the outer nodes as *end nodes*, and sometimes also as Alice and Bob for convenience,
and the in between nodes as the *repeater nodes*.
The lines between the nodes represent both an entangling connection and a classical connection,
as introduced in the `teleportation example <netsquid.examples.teleportation.html>`_.
The repeaters will use a so-called `entanglement swapping scheme <https://en.wikipedia.org/wiki/Quantum_teleportation>`_ to entangle the end nodes,
which consists of the following steps:

1. generating entanglement with both of its neighbours,
2. measuring its two locally stored qubits in the Bell basis,
3. sending its own measurement outcomes to its right neighbour, and also forwarding on outcomes received from its left neighbour in this way.

Let us create the repeater chain network.
We need to create the *N* nodes, each with a quantum processor, and every pair of nodes
in the chain must be linked using an entangling connection and a classical connection.
In each entangling connection an entangled qubit generating source is available.
A schematic illustration of a repeater and its connections is shown below:

.. aafig::
    :textual:

                     +------------------------------------+
                     |          "Repeater"                |
    -------------+   |       +--------------------+       |   +---
    "Entangling" |   |   qin1| "QuantumProcessor" | qin0  |   |
    "Connection" O---O--->---O                    O---<---O---O
                 |   |       |                    |       |   |
    -------------+   |       +--------------------+       |   +---
                     |                                    |
    -------------+   |                                    |   +---
    "Classical"  |   |  "Forward to R"                    |   |
    "Connection" O---O->-# - - - - - - - - - - - - - - - -O---O
                 |   |"ccon_L"                    "ccon_R"|   |
    -------------+   +------------------------------------+   +---

We will re-use the Connection subclasses
:py:class:`~netsquid.examples.teleportation.EntanglingConnection` and :py:class:`~netsquid.examples.teleportation.ClassicalConnection`
created in the `teleportation tutorial <tutorial.simulation.html>`_ and
use the following function to create quantum processors for each node:

.. literalinclude:: ../../netsquid/examples/repeater_chain.py
    :pyobject: create_qprocessor

We create a network component and add the nodes and connections to it.
This way we can easily keep track of all our components in the network, which will be useful when collecting data later.

.. literalinclude:: ../../netsquid/examples/repeater_chain.py
    :pyobject: setup_network

We have used a custom noise model in this example, which helps to exaggerate the effectiveness of the repeater chain.

.. literalinclude:: ../../netsquid/examples/repeater_chain.py
    :pyobject: FibreDepolarizeModel

The next step is to setup the protocols.
To easily manage all the protocols, we add them as subprotocols of one main protocol.
In this way, we can start them at the same time, and the main protocol will stop when all subprotocols have finished.

.. literalinclude:: ../../netsquid/examples/repeater_chain.py
    :pyobject: setup_repeater_protocol

The definition of the swapping subprotocol that will run on each repeater node:

.. literalinclude:: ../../netsquid/examples/repeater_chain.py
    :pyobject: SwapProtocol

The definition of the correction subprotocol responsible for applying the classical corrections at Bob:

.. literalinclude:: ../../netsquid/examples/repeater_chain.py
    :pyobject: CorrectProtocol

Note that for the corrections we only need to apply each of the :math:`X` or :math:`Z` operators maximally once.
This is due to the anti-commutativity and self-inverse of these Pauli matrices, i.e.
:math:`ZX = -XZ` and :math:`XX=ZZ=I`, which allows us to cancel repeated occurrences up to a global phase (-1).
The program that executes the correction on Bob's quantum processor is:

.. literalinclude:: ../../netsquid/examples/repeater_chain.py
    :pyobject: SwapCorrectProgram

With our network and protocols ready, we can add a data collector to define when and which data we want to collect.
We can wait for the signal sent by the *CorrectProtocol* when it finishes, and if it has, compute the fidelity of the
qubits at Alice and Bob with respect to the expected Bell state.
Using our network and main protocol we can easily find Alice and the CorrectionProtocol as subcomponents.

.. literalinclude:: ../../netsquid/examples/repeater_chain.py
    :pyobject: setup_datacollector

We want to run the experiment for multiple numbers of nodes and distances.
Let us first define a function to run a single simulation:

.. literalinclude:: ../../netsquid/examples/repeater_chain.py
    :pyobject: run_simulation

Finally we will run multiple simulations and plot them in a figure.

.. literalinclude:: ../../netsquid/examples/repeater_chain.py
    :pyobject: create_plot

Using the ket vector formalism produces the following figure:

.. image:: ../_static/rep_chain_example_ket.png
    :width: 600px
    :align: center

Because the quantum states don't have an opportunity to grow very large in our simulation,
it is also possible for this simple example to improve on our results using the density matrix formalism.
Instead of running the simulation for 2000 iterations, it is now sufficient to run it for only a few.
As we see in the figure below, the error-bars now become negligible:

.. image:: ../_static/rep_chain_example_dm.png
    :width: 600px
    :align: center

"""
import logging
import pandas
import pydynaa
import numpy as np
import random

import netsquid as ns
from netsquid.qubits import ketstates as ks
from netsquid.components import Message, QuantumProcessor, QuantumProgram, PhysicalInstruction, Clock, Component
from netsquid.components.models.qerrormodels import DepolarNoiseModel, DephaseNoiseModel, QuantumErrorModel
from netsquid.components.instructions import INSTR_MEASURE_BELL, INSTR_X, INSTR_Z
from netsquid.nodes import Node, Network, Connection
from netsquid.protocols import LocalProtocol, NodeProtocol, Signals, Protocol
from netsquid.util.datacollector import DataCollector
from netsquid.examples.teleportation import ClassicalConnection
from netsquid.components.qsource import QSource, SourceStatus
from netsquid.qubits.state_sampler import StateSampler
from netsquid.components.qchannel import QuantumChannel
from netsquid.components.models.delaymodels import FibreDelayModel, FixedDelayModel

__all__ = [
    "SwapProtocol",
    "SwapCorrectProgram",
    "CorrectProtocol",
    "FibreDepolarizeModel",
    "create_qprocessor",
    "setup_network",
    "setup_repeater_protocol",
    "setup_datacollector",
    "run_simulation",
    "create_plot",
]


class SwapProtocol(NodeProtocol):
    """Perform Swap on a repeater node.

    Parameters
    ----------
    node : :class:`~netsquid.nodes.node.Node` or None, optional
        Node this protocol runs on.
    name : str
        Name of this protocol.

    """
    _bsm_op_indices = [(0, 0), (0, 1), (1, 0), (1, 1)]

    def __init__(self, node, name, oracle, pfail):
        super().__init__(node, name)
        self._qmem_input_port_l = self.node.qmemory.ports["qin1"]
        self._qmem_input_port_r = self.node.qmemory.ports["qin0"]
        self.pfail = pfail
        self.oracle = oracle
        self._program = QuantumProgram(num_qubits=2)
        q1, q2 = self._program.get_qubit_indices(num_qubits=2)
        self._program.apply(INSTR_MEASURE_BELL, [q1, q2], output_key="m", inplace=False)

    def run(self):
        while True:
            # Wait for the qubits to arrive
            yield (self.await_port_input(self._qmem_input_port_l) &
                   self.await_port_input(self._qmem_input_port_r))

            logging.debug(f"{ns.sim_time():.1f}: {self.node.name} all qubits received")
            # Randomly drop incoming qubits and notify the oracle about this
            link_states = dict()
            for port in ["qin0", "qin1"]:
                link_states[port] = random.random() >= self.pfail
            self.oracle.add_link_states(self.node, link_states)
            
            # Wait for the oracle to take its decisions
            yield self.await_signal(self.oracle, Signals.SUCCESS)

            # Perform Bell measurement
            swap_possible = True
            for v in link_states.values():
                swap_possible &= v
            if swap_possible:
                logging.debug(f"{ns.sim_time():.1f}: {self.node.name} ready to swap")
                self.node.qmemory.execute_program(self._program, qubit_mapping=[1, 0])
                yield self.await_program(self.node.qmemory)
                m, = self._program.output["m"]
                m1, m2 = self._bsm_op_indices[m]
                # Send result to right node on end
                logging.debug(f"{ns.sim_time():.1f}: {self.node.name} sending corrections out")
                self.node.ports["ccon_R"].tx_output(Message([m1, m2], path=0, timeslot=self.oracle.timeslot))
            else:
                logging.debug(f"{ns.sim_time():.1f}: {self.node.name} cannot swap")


class SwapCorrectProgram(QuantumProgram):
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


class CorrectProtocol(NodeProtocol):
    class PathInfo:
        def __init__(self):
            self.x_corr = 0
            self.z_corr = 0
            self.counter = 0

        def incr(self, x_corr, z_corr):
            self.counter += 1
            self.x_corr += x_corr
            self.z_corr += z_corr

    """Perform corrections for a swap on an end-node.

    Parameters
    ----------
    node : :class:`~netsquid.nodes.node.Node` or None, optional
        Node this protocol runs on.

    """
    def __init__(self, node, oracle):
        super().__init__(node, "CorrectProtocol")
        self.oracle = oracle
        self._program = SwapCorrectProgram()
        self.rx_messages = dict()
        self.timeslot = 0

    def run(self):
        while True:
            yield self.await_port_input(self.node.ports["ccon_L"])
            message = self.node.ports["ccon_L"].rx_input()
            if message is None or len(message.items) != 2:
                continue

            # Retrieve the message's metadata set by the transmitter
            timeslot = message.meta['timeslot']
            path = message.meta['path']
            logging.debug(f"{ns.sim_time():.1f}: {self.node.name} received corrections for timeslot {timeslot} path {path}")

            # If we get timeslot greater than what we have, this means that
            # the last one is expired, so we can clean all pending data
            if timeslot > self.timeslot:
                self.timeslot = timeslot
                self.rx_messages.clear()

            # Count how many messages have been received and check if they
            # are enough for the given path
            if path not in self.rx_messages:
                self.rx_messages[path] = CorrectProtocol.PathInfo()

            path_info = self.rx_messages[path]
            path_length = self.oracle.path_length(path)

            m0, m1 = message.items
            path_info.incr(m1, m0)
            logging.debug(f"{ns.sim_time():.1f}: {self.node.name} got {path_info.counter} correction ({m0}, {m1}) out of {path_length} needed")
            if path_info.counter == path_length:
                if path_info.x_corr or path_info.z_corr:
                    self._program.set_corrections(path_info.x_corr, path_info.z_corr)
                    self.node.qmemory.execute_program(self._program, qubit_mapping=[1])
                    yield self.await_program(self.node.qmemory)
                self.oracle.success()
                self.send_signal(Signals.SUCCESS)
                logging.debug(f"{ns.sim_time():.1f}: {self.node.name} corrections applied, notifying SUCCESS")

class FibreDepolarizeModel(QuantumErrorModel):
    """Custom non-physical error model used to show the effectiveness
    of repeater chains.

    The default values are chosen to make a nice figure,
    and don't represent any physical system.

    Parameters
    ----------
    p_depol_init : float, optional
        Probability of depolarization on entering a fibre.
        Must be between 0 and 1. Default 0.009
    p_depol_length : float, optional
        Probability of depolarization per km of fibre.
        Must be between 0 and 1. Default 0.025

    """
    def __init__(self, p_depol_init=0.009, p_depol_length=0.025):
        super().__init__()
        self.properties['p_depol_init'] = p_depol_init
        self.properties['p_depol_length'] = p_depol_length
        self.required_properties = ['length']

    def error_operation(self, qubits, delta_time=0, **kwargs):
        """Uses the length property to calculate a depolarization probability,
        and applies it to the qubits.

        Parameters
        ----------
        qubits : tuple of :obj:`~netsquid.qubits.qubit.Qubit`
            Qubits to apply noise to.
        delta_time : float, optional
            Time qubits have spent on a component [ns]. Not used.

        """
        for qubit in qubits:
            prob = 1 - (1 - self.properties['p_depol_init']) * np.power(
                10, - kwargs['length']**2 * self.properties['p_depol_length'] / 10)
            ns.qubits.depolarize(qubit, prob=prob)

class Oracle(Protocol):
    def __init__(self, nodes):
        super().__init__(name="Oracle")

        self.node_names = [x.name for x in nodes]
        self.pending_nodes = set(self.node_names)
        self.src_node = None
        self.dst_node = None
        self.timeslot = 0
        self.num_attempts = 0
        self.num_successful = 0

    def set_src_node(self, node):
        logging.debug(f"{ns.sim_time():.1f}: {self.name} {node.name} is the SRC node")
        self.src_node = node.name

    def set_dst_node(self, node):
        logging.debug(f"{ns.sim_time():.1f}: {self.name} {node.name} is the DST node")
        self.dst_node = node.name

    def add_link_states(self, node, link_states):
        assert node.name in self.pending_nodes
        assert self.src_node is not None
        assert self.dst_node is not None

        logging.debug((f"{ns.sim_time():.1f}: {self.name} received "
              f"from node {node.name} link states {link_states}"))

        self.pending_nodes.remove(node.name)  

        if self.pending_nodes == set([self.src_node, self.dst_node]):
            # Send signal to all waiting nodes that they should move on
            self.send_signal(Signals.SUCCESS)
            self.num_attempts += 1

            # This is a new timeslot
            self.timeslot += 1

            # Wait for all nodes again
            self.pending_nodes = set(self.node_names)

    def success(self):
        self.num_successful += 1

    def path_length(self, path):
        if path != 0:
            raise Exception(f"Unknown path {path}")
        return len(self.node_names) - 2

class PassThroughProtocol(Protocol):
    def __init__(self, name, in_port, out_port):
        self.name = name
        self.in_port = in_port
        self.out_port = out_port

    def run(self):
        while True:
            yield self.await_port_input(self.in_port)
            message = self.in_port.rx_input()
            if message is not None:
                logging.debug(f"{ns.sim_time():.1f}: message received by {self.name}: {message}")
            self.out_port.tx_output(message)

class MyEntanglingConnection(Connection):
    """A connection that generates entanglement.

    Consists of a midpoint holding a quantum source that connects to
    outgoing quantum channels.

    Parameters
    ----------
    length : float
        End to end length of the connection [km].
    source_frequency : float
        Frequency with which midpoint entanglement source generates entanglement [Hz].
    name : str, optional
        Name of this connection.

    """

    def __init__(self, length, source_frequency, name="MyEntanglingConnection"):

        super().__init__(name=name)
        qsource = QSource(f"qsource_{name}", StateSampler([ks.b00], [1.0]), num_ports=2,
                          timing_model=FixedDelayModel(delay=1e9 / source_frequency),
                          status=SourceStatus.EXTERNAL)
        self.add_subcomponent(qsource, name="qsource")

        clock = Clock("clock", frequency=source_frequency, max_ticks=-1, start_delay=0)
        self.add_subcomponent(clock, name="clock")

        pass_through_a = Component(name=name + "_PassThroughA", port_names=["in_port", "out_port"])
        self.add_subcomponent(pass_through_a, name="passthrough_a")

        pass_through_b = Component(name=name + "_PassThroughB", port_names=["in_port", "out_port"])
        self.add_subcomponent(pass_through_b, name="passthrough_b")

        self.pass_through_protocol_a = PassThroughProtocol(
            pass_through_a.name + "_proto",
            pass_through_a.ports["in_port"], pass_through_a.ports["out_port"])
        self.pass_through_protocol_b = PassThroughProtocol(
            pass_through_b.name + "_proto",
            pass_through_b.ports["in_port"], pass_through_b.ports["out_port"])
        self.pass_through_protocol_a.start()
        self.pass_through_protocol_b.start()

        clock.ports["cout"].connect(qsource.ports["trigger"])
        clock.start()

        qchannel_c2a = QuantumChannel("qchannel_C2A", length=length / 2,
                                      models={"delay_model": FibreDelayModel()})
        qchannel_c2b = QuantumChannel("qchannel_C2B", length=length / 2,
                                      models={"delay_model": FibreDelayModel()})
        # Add channels and forward quantum channel output to external port output:
        self.add_subcomponent(qchannel_c2a, forward_output=[("A", "recv")])
        self.add_subcomponent(qchannel_c2b, forward_output=[("B", "recv")])
        # Connect qsource output to quantum channel input:
        # qsource.ports["qout0"].connect(qchannel_c2a.ports["send"])
        # qsource.ports["qout1"].connect(qchannel_c2b.ports["send"])
        qsource.ports["qout0"].connect(pass_through_a.ports["in_port"])
        pass_through_a.ports["out_port"].connect(qchannel_c2a.ports["send"])
        qsource.ports["qout1"].connect(pass_through_b.ports["in_port"])
        pass_through_b.ports["out_port"].connect(qchannel_c2b.ports["send"])

def create_qprocessor(name):
    """Factory to create a quantum processor for each node in the repeater chain network.

    Has two memory positions and the physical instructions necessary for teleportation.

    Parameters
    ----------
    name : str
        Name of the quantum processor.

    Returns
    -------
    :class:`~netsquid.components.qprocessor.QuantumProcessor`
        A quantum processor to specification.

    """
    noise_rate = 200
    gate_duration = 1
    gate_noise_model = DephaseNoiseModel(noise_rate)
    mem_noise_model = DepolarNoiseModel(noise_rate)
    physical_instructions = [
        PhysicalInstruction(INSTR_X, duration=gate_duration,
                            q_noise_model=gate_noise_model),
        PhysicalInstruction(INSTR_Z, duration=gate_duration,
                            q_noise_model=gate_noise_model),
        PhysicalInstruction(INSTR_MEASURE_BELL, duration=gate_duration),
    ]
    qproc = QuantumProcessor(name, num_positions=2, fallback_to_nonphysical=False,
                             mem_noise_models=[mem_noise_model] * 2,
                             phys_instructions=physical_instructions)
    return qproc


def setup_network(num_nodes, node_distance, source_frequency):
    """Setup repeater chain network.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the network, at least 3.
    node_distance : float
        Distance between nodes [km].
    source_frequency : float
        Frequency at which the sources create entangled qubits [Hz].

    Returns
    -------
    :class:`~netsquid.nodes.network.Network`
        Network component with all nodes and connections as subcomponents.

    """
    if num_nodes < 3:
        raise ValueError(f"Can't create repeater chain with {num_nodes} nodes.")
    network = Network("Repeater_chain_network")

    # Create nodes with quantum processors
    nodes = []
    for i in range(num_nodes):
        # Prepend leading zeros to the number
        num_zeros = int(np.log10(num_nodes)) + 1
        nodes.append(Node(f"Node_{i:0{num_zeros}d}", qmemory=create_qprocessor(f"qproc_{i}")))
    network.add_nodes(nodes)
    # Create quantum and classical connections:
    for i in range(num_nodes - 1):
        node, node_right = nodes[i], nodes[i+1]
        # Create quantum connection
        qconn = MyEntanglingConnection(name=f"qconn_{i}-{i+1}", length=node_distance,
                                       source_frequency=source_frequency)
        # Add a noise model which depolarizes the qubits exponentially
        # depending on the connection length
        for channel_name in ['qchannel_C2A', 'qchannel_C2B']:
            qconn.subcomponents[channel_name].models['quantum_noise_model'] =\
                FibreDepolarizeModel()
        port_name, port_r_name = network.add_connection(
            node, node_right, connection=qconn, label="quantum")
        # Forward qconn directly to quantum memories for right and left inputs:
        node.ports[port_name].forward_input(node.qmemory.ports["qin0"])  # R input
        node_right.ports[port_r_name].forward_input(
            node_right.qmemory.ports["qin1"])  # L input
        # Create classical connection
        cconn = ClassicalConnection(name=f"cconn_{i}-{i+1}", length=node_distance)
        port_name, port_r_name = network.add_connection(
            node, node_right, connection=cconn, label="classical",
            port_name_node1="ccon_R", port_name_node2="ccon_L")
        # Forward cconn to right most node
        if "ccon_L" in node.ports:
            node.ports["ccon_L"].bind_input_handler(
                lambda message, _node=node: _node.ports["ccon_R"].tx_output(message))

    # Create oracle
    oracle = Oracle(nodes)
    oracle.set_src_node(nodes[0])
    oracle.set_dst_node(nodes[-1])

    return (network, oracle)

def setup_repeater_protocol(network, oracle, pfail):
    """Setup repeater protocol on repeater chain network.

    Parameters
    ----------
    network : :class:`~netsquid.nodes.network.Network`
        Repeater chain network to put protocols on.

    Returns
    -------
    :class:`~netsquid.protocols.protocol.Protocol`
        Protocol holding all subprotocols used in the network.

    """
    protocol = LocalProtocol(nodes=network.nodes)
    # Add SwapProtocol to all repeater nodes. Note: we use unique names,
    # since the subprotocols would otherwise overwrite each other in the main protocol.
    nodes = [network.nodes[name] for name in sorted(network.nodes.keys())]
    for node in nodes[1:-1]:
        subprotocol = SwapProtocol(node=node, name=f"Swap_{node.name}", oracle=oracle, pfail=pfail)
        protocol.add_subprotocol(subprotocol)
    # Add CorrectProtocol to Bob
    subprotocol = CorrectProtocol(nodes[-1], oracle=oracle)
    protocol.add_subprotocol(subprotocol)
    # Add oracle
    protocol.add_subprotocol(oracle)
    return protocol


def setup_datacollector(network, protocol):
    """Setup the datacollector to calculate the fidelity
    when the CorrectionProtocol has finished.

    Parameters
    ----------
    network : :class:`~netsquid.nodes.network.Network`
        Repeater chain network to put protocols on.

    protocol : :class:`~netsquid.protocols.protocol.Protocol`
        Protocol holding all subprotocols used in the network.

    Returns
    -------
    :class:`~netsquid.util.datacollector.DataCollector`
        Datacollector recording fidelity data.

    """
    # Ensure nodes are ordered in the chain:
    nodes = [network.nodes[name] for name in sorted(network.nodes.keys())]

    def calc_fidelity(evexpr):
        qubit_a, = nodes[0].qmemory.peek([0])
        qubit_b, = nodes[-1].qmemory.peek([1])
        fidelity = ns.qubits.fidelity([qubit_a, qubit_b], ks.b00, squared=True)
        return {"fidelity": fidelity}

    dc = DataCollector(calc_fidelity, include_entity_name=False)
    dc.collect_on(pydynaa.EventExpression(source=protocol.subprotocols['CorrectProtocol'],
                                          event_type=Signals.SUCCESS.value))
    return dc

def run_simulation(num_nodes=4, node_distance=20, num_iters=100, pfail=0, seed=42):
    """Run the simulation experiment and return the collected data.

    Parameters
    ----------
    num_nodes : int, optional
        Number nodes in the repeater chain network. At least 3. Default 4.
    node_distance : float, optional
        Distance between nodes, larger than 0. Default 20 [km].
    num_iters : int, optional
        Number of simulation runs. Default 100.

    Returns
    -------
    :class:`pandas.DataFrame`
        Dataframe with recorded fidelity data.

    """
    logging.info(f"starting simulation: num_nodes = {num_nodes}, distance = {node_distance} km, messages = {num_iters}, prob. entang. failure = {pfail}")
    ns.sim_reset()
    ns.set_random_state(seed=seed)
    random.seed(seed)
    est_runtime = (0.5 + num_nodes - 1) * node_distance * 5e3
    logging.debug(f"estimated end-to-end delay = {est_runtime} ns")
    (network, oracle) = setup_network(num_nodes, node_distance=node_distance,
                                      source_frequency=1e9 / est_runtime)
    protocol = setup_repeater_protocol(network, oracle, pfail)
    dc = setup_datacollector(network, protocol)
    protocol.start()
    ns.sim_run(est_runtime * num_iters)
    return (dc.dataframe,
            protocol.subprotocols['Oracle'].num_attempts,
            protocol.subprotocols['Oracle'].num_successful)

def create_plot(num_iters=2000):
    """Run the simulation for multiple nodes and distances and show them in a figure.

    Parameters
    ----------

    num_iters : int, optional
        Number of iterations per simulation configuration.
        At least 1. Default 2000.
    """
    from matplotlib import pyplot as plt
    _, ax = plt.subplots()
    data_rates = []
    for distance in [10, 30, 50]:
        for pfail in [0, 0.1]:
            data = pandas.DataFrame()
            # for num_node in range(8, 9):
            for num_node in [3, 8]:
                res = run_simulation(num_nodes=num_node,
                                     node_distance=distance / num_node,
                                     num_iters=num_iters,
                                     pfail=pfail)
                assert len(res) == 3
                data[num_node] = res[0]['fidelity']
                assert res[1] > 0
                data_rates.append({
                    'distance': distance,
                    'pfail': pfail,
                    'num_node': num_node,
                    'attempts': res[1],
                    'successful': res[2],
                    'success_prob': res[2]/res[1]
                    })

            # For errorbars we use the standard error of the mean (sem)
            data = data.agg(['mean', 'sem']).T.rename(columns={'mean': 'fidelity'})
            data.plot(y='fidelity', yerr='sem', label=f"{distance} km, p = {pfail}", ax=ax)
    plt.xlabel("number of nodes")
    plt.ylabel("fidelity")
    plt.title("Repeater chain with different total lengths\nand entanglement failure probabilities")
    plt.show(block=False)

    # Plot the probabilities
    df_rates = pandas.DataFrame(data_rates)

    metrics = ['successful', 'success_prob']

    for metric in metrics:
        _, ax = plt.subplots()
        
        for distance in set(df_rates['distance']):
            for pfail in set(df_rates['pfail']):
                df_rates.loc[(df_rates['distance'] == distance) & (df_rates['pfail'] == pfail)].\
                    plot(x='num_node', y=metric, label=f"{distance} km, p = {pfail}", ax=ax)

        plt.xlabel("number of nodes")
        plt.ylabel(metric)
        plt.title("Repeater chain with different total lengths\nand entanglement failure probabilities")
        plt.show(block=(metric == metrics[-1]))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ns.set_qstate_formalism(ns.QFormalism.DM)

    num_timeslots = 100

    create_plot(num_iters=num_timeslots)

    # distance = 10 # km
    # seed = 1
    # pfail = 0.25

    # df = run_simulation(num_nodes=9,
    #                     node_distance=distance,
    #                     num_iters=num_timeslots,
    #                     pfail=pfail,
    #                     seed=seed)

    # logging.info(df)