"""
Example showing how to use quantum programs in NetSquid, see:

https://docs.netsquid.org/latest-release/tutorial.quantumprocessor.html

+-------------+       +---------------+       +--------------+
|             |  DC   |               |  DC   |              |
| create_node +-------+ entangle_node +-------+ measure_node |
|             |       |               |       |              |
+-------------+       +---------------+       +--------------+

Roles of nodes:

- create_node: create two qubits |0> periodically and send them to entangle_node

- entangle_node: entangle the two qubits as a Bell b_00 state, send to measure_node

- mesure_node: measure the fidelity of the 2-qubit state received

The nodes are connected via QuantumChannels with FibreLossModel, hence the
qubits may be lost with some probability.

The are three possible scenarios simulated, which only differ on entangle_node:

- ideal: quantum operations are instantaneous and ideal

- qmemory: entangle_node has a quantum memory with DepolarNoiseModel, which
  takes some time to transfer the qubits from channel to memory

- qpu: entangle_node executes the entanglement operations as a quantum program
"""

import math
import random
import logging
import netsquid as ns
from netsquid.protocols import NodeProtocol
from netsquid.components import QuantumChannel, QuantumMemory
from netsquid.nodes import Node, DirectConnection
from netsquid.qubits import qubitapi as qapi, ketstates
from netsquid.components.models import DepolarNoiseModel
from netsquid.components.models.qerrormodels import FibreLossModel

class CreateProtocol(NodeProtocol):
    create_period = 1000

    def run(self):
        logging.info(f"{ns.sim_time():.1f}: CreateProtocol started on {self.node.name}")
        while True:
            yield self.await_timer(self.create_period)
            self.node.ports["to_entangle"].tx_output(qapi.create_qubits(2))

class EntangleProtocolIdeal(NodeProtocol):
    def run(self):
        logging.info(f"{ns.sim_time():.1f}: EntangleProtocolIdeal started on {self.node.name}")
        while True:
            yield self.await_port_input(self.node.ports["from_create"])
            qubits = self.node.ports["from_create"].rx_input().items
            if len(qubits) != 2:
                continue
            ns.qubits.operate(qubits[0], ns.H)
            ns.qubits.operate(qubits, ns.CX)
            self.node.ports["to_measure"].tx_output(qubits)

class EntangleProtocolQMemory(NodeProtocol):
    memory_transfer_delay = 10

    def run(self):
        logging.info(f"{ns.sim_time():.1f}: EntangleProtocolQpu started on {self.node.name}")
        while True:
            # wait for the qubits to arrive from the channel
            yield self.await_port_input(self.node.qmemory.ports["qin"])

            if self.node.qmemory.num_used_positions != 2:
                continue
            
            # wait a random time to simulate transfer delay of memory components
            yield self.await_timer(random.uniform(
                self.memory_transfer_delay * 0.5,
                self.memory_transfer_delay * 1.5))

            # entangle the two qubits
            qubits = self.node.qmemory.peek(positions=[0, 1])
            ns.qubits.operate(qubits[0], ns.H)
            ns.qubits.operate(qubits, ns.CX)
            self.node.qmemory.pop(positions=[0, 1])

class EntangleProtocolQpu(NodeProtocol):
    memory_transfer_delay = 10

    def run(self):
        logging.info(f"{ns.sim_time():.1f}: EntangleProtocolQpu started on {self.node.name}")
        while True:
            # wait for the qubits to arrive from the channel
            yield self.await_port_input(self.node.qmemory.ports["qin"])

            if self.node.qmemory.num_used_positions != 2:
                continue
            
            # wait a random time to simulate transfer delay of memory components
            yield self.await_timer(random.uniform(
                self.memory_transfer_delay * 0.5,
                self.memory_transfer_delay * 1.5))

            # entangle the two qubits
            qubits = self.node.qmemory.peek(positions=[0, 1])
            ns.qubits.operate(qubits[0], ns.H)
            ns.qubits.operate(qubits, ns.CX)
            self.node.qmemory.pop(positions=[0, 1])

class MeasureProtocol(NodeProtocol):
    def __init__(self, node):
        super().__init__(node)
        self.fidelity = 0.
        self.received = 0

    def run(self):
        logging.info(f"{ns.sim_time():.1f}: MeasureProtocol started on {self.node.name}")
        while True:
            yield self.await_port_input(self.node.ports["from_entangle"])
            qubits = self.node.ports["from_entangle"].rx_input().items
            if len(qubits) != 2:
                continue
            fidelity = ns.qubits.fidelity(qubits, ketstates.b00, squared=True)
            logging.info(f"{ns.sim_time():.1f}: {self.node.name} fidelity {fidelity:.3f}")

            self.received += 1
            self.fidelity += fidelity

def run_replication(scenario, seed, depolar_rate):
    # Configuration
    sim_duration = 10000 # ns
    length = 1e-9 # km
    ns.sim_reset()
    ns.set_random_state(seed=seed)
    random.seed(seed)
    ns.set_qstate_formalism(ns.QFormalism.DM)
    logging.basicConfig(level=logging.WARNING)

    # Check configuration
    assert scenario in ['ideal', 'qmemory', 'qpu']

    # Create nodes
    create_node = Node("CreateNode", port_names=["to_entangle"])
    if scenario == 'ideal':
        entangle_node = Node("EntangleNode", port_names=["from_create", "to_measure"])
    elif scenario == 'qmemory' or scenario == 'qpu':
        noise_model = DepolarNoiseModel(depolar_rate=depolar_rate)
        entangle_qmem = QuantumMemory(
            name="DepolarMemory",
            num_positions=2,
            memory_noise_models=[noise_model] * 2)
        entangle_node = Node(
            "EntangleNode",
            port_names=["from_create", "to_measure"],
            qmemory=entangle_qmem)
        entangle_node.ports['from_create'].forward_input(entangle_node.qmemory.ports['qin'])
        entangle_node.qmemory.ports['qout'].forward_output(entangle_node.ports['to_measure'])
    measure_node = Node("MeasureNode", port_names=["from_entangle"])

    # Create connections
    channel_loss_model = FibreLossModel(p_loss_init=0.2, p_loss_length=0.1)
    create_entangle_dc = DirectConnection(
        "Create2Entangle",
        QuantumChannel(f"Channel_LR_C2E", length=length, models={'quantum_loss_model': channel_loss_model}),
        QuantumChannel(f"Channel_RL_C2E", length=length, models={'quantum_loss_model': channel_loss_model}))
    entangle_measure_dc = DirectConnection(
        "Entangle2Measure",
        QuantumChannel(f"Channel_LR_E2M", length=length, models={'quantum_loss_model': channel_loss_model}),
        QuantumChannel(f"Channel_RL_E2M", length=length, models={'quantum_loss_model': channel_loss_model}))

    # Connect nodes through connections
    create_node.ports["to_entangle"].connect(create_entangle_dc.ports["A"])
    entangle_node.ports["from_create"].connect(create_entangle_dc.ports["B"])
    entangle_node.ports["to_measure"].connect(entangle_measure_dc.ports["A"])
    measure_node.ports["from_entangle"].connect(entangle_measure_dc.ports["B"])

    # Create protocols
    protocols = [
        MeasureProtocol(measure_node),
        CreateProtocol(create_node)
    ]

    if scenario == 'ideal':
        protocols.append(EntangleProtocolIdeal(entangle_node))
    elif scenario == 'qmemory':
        protocols.append(EntangleProtocolQMemory(entangle_node))
    elif scenario == 'qpu':
        protocols.append(EntangleProtocolQpu(entangle_node))

    # Start all protocols
    for protocol in protocols:
        protocol.start()

    # Run simulator
    ns.sim_run(duration=sim_duration)

    return (protocols[0].fidelity / protocols[0].received, protocols[0].received)

# main loop
for s in ['ideal', 'qmemory', 'qpu']:
    for d in [1e6, 1e7, 1e8]:
        for i in range(1):
            logging.info(f"replication {i}")
            res = run_replication(scenario=s, seed=i, depolar_rate=d)
            print(f'{s} depol-rate 10^{math.log10(d):.0f} {i} {res[0]:.3f}, {res[1]}')