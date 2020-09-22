"""
Example showing how to use a QSource, i.e., a generator of qubits.

The QSource sends to two nodes, with different distance from the emitter:
the first node does a measurement, while the other checks the fidelity.

This is a modification of the "Quantum teleportation using components" example
from the NetSquid documentation:
https://docs.netsquid.org/latest-release/tutorial.components.html
"""

import random
import pydynaa
import netsquid as ns
from netsquid.components import Channel
from netsquid.components.component import Port
from netsquid.components.qchannel import QuantumChannel
from netsquid.components.qmemory import QuantumMemory
from netsquid.components.models.delaymodels import FibreDelayModel, FixedDelayModel
from netsquid.qubits.state_sampler import StateSampler
from netsquid.components.qsource import QSource, SourceStatus
from netsquid.components.models.qerrormodels import DepolarNoiseModel, DephaseNoiseModel
import netsquid.qubits.ketstates as ks

class ReceivingNode(pydynaa.Entity):
    counter = 0
    depolar_rate = 1e7  # depolarization rate of waiting qubits [Hz]
    dephase_rate = 1e7  # dephasing noise rate of waiting qubits [Hz]

    def __init__(self):
        self.name = f'RingNode#{self.counter}'
        ReceivingNode.counter += 1

        noise_model = DepolarNoiseModel(depolar_rate=0.1, time_independent=True)
        # noise_model = DepolarNoiseModel(depolar_rate=self.depolar_rate)
        # noise_model = DephaseNoiseModel(dephase_rate=self.dephase_rate)
        self.qmemory = QuantumMemory("NodeMemory", num_positions=1,
                                     memory_noise_models=[noise_model])
        self._wait(pydynaa.EventHandler(self._handle_input_qubit),
                   entity=self.qmemory.ports["qin0"],
                   event_type=Port.evtype_input)
        self.qmemory.ports["qin0"].notify_all_input = True

    def _handle_input_qubit(self, event):
        if self.name == 'RingNode#0':
            [m], [prob] = self.qmemory.measure(
                positions=[0], observable=ns.Z, discard=True)
            labels_z = ("|0>", "|1>")
            print(f"{ns.sim_time():.1f}: {self.name} qubit measured "
                  f"{labels_z[m]} with probability {prob:.2f}")
        else:
            # Pop qubit from memory and check fidelity
            self.qmemory.operate(ns.X, positions=[0])
            qubit = self.qmemory.pop(positions=[0])
            # print(qubit[0].qstate)
            fidelity = ns.qubits.fidelity(qubit[0], ns.s1, squared=True)
            print(f"{ns.sim_time():.1f}: {self.name} received qubit, "
                f"fidelity = {fidelity:.3f}")

# Configuration
seed = 42
length=4e-3 # km
ns.set_qstate_formalism(ns.QFormalism.DM)
ns.sim_reset()

# Create a QSource with two ports emitting qubits every 50 ns
state_sampler = StateSampler([ks.s00], [1.0])
qsource = QSource("QSource", state_sampler, num_ports=2,
                  timing_model=FixedDelayModel(delay=50),
                  status=SourceStatus.INTERNAL)

# Create two nodes
nodes = [ReceivingNode(), ReceivingNode()]

# Create the quantum channels connecting the QSource with the two nodes
qchannels = [QuantumChannel("qchannel0", length=length / 2,
                            models={"delay_model": FibreDelayModel()}),
             QuantumChannel("qchannel0", length=length,
                            models={"delay_model": FibreDelayModel()})]               
for i in [0, 1]:
    qsource.ports[f'qout{i}'].connect(qchannels[i].ports['send'])
    nodes[i].qmemory.ports['qin0'].connect(qchannels[i].ports['recv'])

# Run simulator
ns.set_random_state(seed=seed)
stats = ns.sim_run(duration=200)
print(stats)