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
  on a noisy quantum processor, with depolarizing and dephasing memory; also,
  the instructions are modeled with > 0 delay

Output metrics:

- fidelity, comparing the measured quantum state w.r.t. ideal b_00

- delay, defined as the time between when the qubits are created and when
  their fidelity is measured; this includes: the fibre propagation delay, the
  memory transfer delay (with qmemory and qpu), and the processing delay
  (with qpu only)

- the number of qubits lost, due to channel propagation; such qubits do not
  reach the final destination, hence they are not accounted in the other metrics
"""

import sys
import pandas
import math
import random
import logging
import pydynaa

import netsquid as ns
from netsquid.protocols import NodeProtocol
from netsquid.components import QuantumChannel, QuantumMemory, QuantumProgram, QuantumProcessor, PhysicalInstruction
from netsquid.nodes import Node, DirectConnection
from netsquid.protocols.protocol import Signals
from netsquid.qubits import qubitapi as qapi, ketstates
from netsquid.components.models import DepolarNoiseModel, DephaseNoiseModel
from netsquid.components.models.delaymodels import FibreDelayModel
from netsquid.components.models.qerrormodels import FibreLossModel
from netsquid.util.datacollector import DataCollector
import netsquid.components.instructions as instr

class CreateProtocol(NodeProtocol):
    create_period = 100000

    def run(self):
        logging.info(f"{ns.sim_time():.1f}: CreateProtocol started on {self.node.name}")
        self.counter = 0
        while True:
            yield self.await_timer(self.create_period)
            qubits = qapi.create_qubits(2)
            self.node.ports["to_entangle"].tx_output(qubits)
            self.last_time = ns.sim_time()
            self.counter += 1
            logging.info(f"{ns.sim_time():.1f}: CreateProtocol sending out qubits")

class EntangleProtocol(NodeProtocol):
    def __init__(self, node, memory_tdelay):
        super().__init__(node)
        self.memory_tdelay = memory_tdelay

    def memory_tdelay_wait(self):
        """Wait a random time to simulate transfer delay of memory components"""

        wait_time = random.uniform(self.memory_tdelay * 0.5,
                                   self.memory_tdelay * 1.5)
        return self.await_timer(wait_time)

class EntangleProtocolIdeal(EntangleProtocol):
    def run(self):
        logging.info(f"{ns.sim_time():.1f}: EntangleProtocolIdeal started on {self.node.name}")
        while True:
            yield self.await_port_input(self.node.ports["from_create"])
            logging.info(f"{ns.sim_time():.1f}: EntangleProtocolIdeal received qubits")
            qubits = self.node.ports["from_create"].rx_input().items
            if len(qubits) != 2:
                continue
            ns.qubits.operate(qubits[0], ns.H)
            ns.qubits.operate(qubits, ns.CX)
            self.node.ports["to_measure"].tx_output(qubits)

class EntangleProtocolQMemory(EntangleProtocol):
    def run(self):
        logging.info(f"{ns.sim_time():.1f}: EntangleProtocolQpu started on {self.node.name}")
        while True:
            # wait for the qubits to arrive from the channel

            yield self.await_port_input(self.node.qmemory.ports["qin"])

            if self.node.qmemory.num_used_positions != 2:
                continue
            
            yield self.memory_tdelay_wait()
           
            # entangle the two qubits
            qubits = self.node.qmemory.peek(positions=[0, 1])
            ns.qubits.operate(qubits[0], ns.H)
            ns.qubits.operate(qubits, ns.CX)
            self.node.qmemory.pop(positions=[0, 1])

class EntangleProgram(QuantumProgram):
    default_num_qubits = 2

    def program(self):
        q1, q2 = self.get_qubit_indices(2)
        self.apply(instr.INSTR_H, q1)
        self.apply(instr.INSTR_CNOT, [q1, q2])
        yield self.run()

class EntangleProtocolQpu(EntangleProtocol):
    def run(self):
        logging.info(f"{ns.sim_time():.1f}: EntangleProtocolQpu started on {self.node.name}")

        prog = EntangleProgram()
        while True:
            # wait for the qubits to arrive from the channel
            yield self.await_port_input(self.node.ports["from_create"])

            if self.node.qmemory.num_used_positions != 2:
                continue

            yield self.memory_tdelay_wait()

            # entangle the two qubits by executing a program on the QPU
            self.node.qmemory.execute_program(prog)
            yield self.await_program(self.node.qmemory)
            self.node.qmemory.pop(positions=[0, 1])

class MeasureProtocol(NodeProtocol):
    def __init__(self, node):
        super().__init__(node)
        self.fidelity = 0.

    def run(self):
        logging.info(f"{ns.sim_time():.1f}: MeasureProtocol started on {self.node.name}")
        while True:
            yield self.await_port_input(self.node.ports["from_entangle"])
            logging.info(f"{ns.sim_time():.1f}: MeasureProtocol received qubits")
            qubits = self.node.ports["from_entangle"].rx_input().items
            if len(qubits) != 2:
                continue

            # Measure fidelity
            self.fidelity = ns.qubits.fidelity(qubits, ketstates.b00, squared=True)
            logging.info(f"{ns.sim_time():.1f}: {self.node.name} fidelity {self.fidelity:.3f}")

            # Notify that the fidelity can be collected
            self.send_signal(Signals.SUCCESS)

def create_memory(depolar_rate):
    noise_model = DepolarNoiseModel(depolar_rate=depolar_rate)
    memory = QuantumMemory(name="DepolarMemory",
                           num_positions=2,
                           memory_noise_models=[noise_model] * 2)
    return memory

def create_processor(depolar_rate, dephase_rate):
    measure_noise_model = DephaseNoiseModel(dephase_rate=dephase_rate,
                                            time_independent=True)
    memory_noise_model = DepolarNoiseModel(depolar_rate=depolar_rate)
    physical_instructions = [
        PhysicalInstruction(instr.INSTR_INIT, duration=3, parallel=True),
        PhysicalInstruction(instr.INSTR_H, duration=1, parallel=True, topology=[0, 1]),
        PhysicalInstruction(instr.INSTR_X, duration=1, parallel=True, topology=[0], q_noise_model=measure_noise_model),
        PhysicalInstruction(instr.INSTR_Z, duration=1, parallel=True, topology=[0]),
        PhysicalInstruction(instr.INSTR_S, duration=1, parallel=True, topology=[0]),
        PhysicalInstruction(instr.INSTR_CNOT, duration=4, parallel=True, topology=[(0, 1)]),
        PhysicalInstruction(instr.INSTR_MEASURE, duration=7, parallel=False, topology=[0],
                            q_noise_model=measure_noise_model, apply_q_noise_after=False),
        PhysicalInstruction(instr.INSTR_MEASURE, duration=7, parallel=False, topology=[1])
    ]
    processor = QuantumProcessor("Qpu",
                                 num_positions=2,
                                 models={'qout_noise_model': measure_noise_model},
                                 mem_noise_models=[memory_noise_model] * 2,
                                 phys_instructions=physical_instructions)

    return processor

def run_replication(scenario, seed, depolar_rate, dephase_rate, memory_tdelay):
    # Configuration
    sim_duration = 1e7 # ns
    length = 1 # km
    ns.sim_reset()
    random.seed(seed)
    ns.set_random_state(seed=seed)
    ns.set_qstate_formalism(ns.QFormalism.DM)

    # Check configuration
    assert scenario in ['ideal', 'qmemory', 'qpu']

    #
    # Create nodes
    #
    
    # node that creates the qubits
    create_node = Node("CreateNode", port_names=["to_entangle"])

    # node that entangles them, depending on the scenario
    if scenario == 'ideal':
        entangle_node = Node("EntangleNode", port_names=["from_create", "to_measure"])

    elif scenario == 'qmemory' or scenario == 'qpu':
        qmemory = create_memory(depolar_rate) \
            if scenario == 'qmemory' \
            else create_processor(depolar_rate=depolar_rate, dephase_rate=dephase_rate)
        entangle_node = Node(
            "EntangleNode",
            port_names=["from_create", "to_measure"],
            qmemory=qmemory)
        entangle_node.ports['from_create'].forward_input(qmemory.ports['qin'])
        qmemory.ports['qout'].forward_output(entangle_node.ports['to_measure'])

    assert entangle_node is not None
    
    # node that does the fidelity measurement
    measure_node = Node("MeasureNode", port_names=["from_entangle"])

    # Create connections
    channel_loss_model = FibreLossModel(p_loss_init=0.1, p_loss_length=0.05)
    create_entangle_dc = DirectConnection(
        "Create2Entangle",
        QuantumChannel(f"Channel_LR_C2E", length=length, models={'quantum_loss_model': channel_loss_model, 'delay_model': FibreDelayModel()}),
        QuantumChannel(f"Channel_RL_C2E", length=length, models={'quantum_loss_model': channel_loss_model, 'delay_model': FibreDelayModel()}))
    entangle_measure_dc = DirectConnection(
        "Entangle2Measure",
        QuantumChannel(f"Channel_LR_E2M", length=length, models={'quantum_loss_model': channel_loss_model, 'delay_model': FibreDelayModel()}),
        QuantumChannel(f"Channel_RL_E2M", length=length, models={'quantum_loss_model': channel_loss_model, 'delay_model': FibreDelayModel()}))

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
        protocols.append(EntangleProtocolIdeal(entangle_node, memory_tdelay))
    elif scenario == 'qmemory':
        protocols.append(EntangleProtocolQMemory(entangle_node, memory_tdelay))
    elif scenario == 'qpu':
        protocols.append(EntangleProtocolQpu(entangle_node, memory_tdelay))

    # Data collection
    def collect_fidelity_data(evexpr):
        protocol = evexpr.triggered_events[-1].source
        delay = ns.sim_time() - protocols[1].last_time
        losses = protocols[1].counter - 1
        protocols[1].counter = 0
        return {"fidelity": protocol.fidelity, "delay": delay, "losses": losses}

    dc = DataCollector(collect_fidelity_data)
    dc.collect_on(pydynaa.EventExpression(source=protocols[0],
                                          event_type=Signals.SUCCESS.value))

    # Start all protocols
    for protocol in protocols:
        protocol.start()

    # Run simulator
    ns.sim_run(duration=sim_duration)

    return dc.dataframe

if __name__ == '__main__':
    # logging.basicConfig(level=logging.INFO)
    # run_replication(scenario='ideal', seed=42, depolar_rate=1e7, dephase_rate=0.2, memory_tdelay=10)
    # run_replication(scenario='qpu', seed=42, depolar_rate=1e7, dephase_rate=0.2, memory_tdelay=10)
    # run_replication(scenario='qpu', seed=43, depolar_rate=1e7, dephase_rate=0.2, memory_tdelay=10)
    # sys.exit(0)

    data = pandas.DataFrame()

    for scenario in ['ideal', 'qmemory', 'qpu']:
        for depolar_rate in [1e6, 1e7, 1e8]:
            for dephase_rate in [0.01, 0.1]:
                for memory_tdelay in [1, 5, 10]:
                    for seed in range(1):
                        logging.info(f"replication {seed}")
                        df = run_replication(scenario=scenario, seed=seed,
                                            depolar_rate=depolar_rate,
                                            dephase_rate=dephase_rate,
                                            memory_tdelay=memory_tdelay)
                        df['scenario'] = scenario
                        df['depolar_rate'] = depolar_rate
                        df['dephase_rate'] = dephase_rate
                        df['memory_tdelay'] = memory_tdelay
                        data = data.append(df)

    print(data.groupby(
        ["depolar_rate", "scenario", "dephase_rate", "memory_tdelay"])['fidelity'].agg(
            fidelity='mean', sem='sem').reset_index())

    print(data.groupby(
        ["depolar_rate", "scenario", "dephase_rate", "memory_tdelay"])['delay'].agg(
            delay='mean', sem='sem').reset_index())

    print(data.groupby(
        ["depolar_rate", "scenario", "dephase_rate", "memory_tdelay"])['losses'].agg(
            losses='sum').reset_index())
