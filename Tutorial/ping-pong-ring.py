"""
Ring of elements, each made of a single-qubit memory and a quantum channel,
passing qubit from one to another until the qubit is received by the
first node.

Structure of any node:

---------+
         |
    qin0 O <------- from recv port of next nodes' channel
         |      +------------+
    qout O ---> O send  recv O ---> to next node's qin0 
         |      +------------+
---------+

When receiving the qubit, stored in the memory from the channel, every
intermediate node measures it with 50% probability; the last node, which is
the same which created the qubit, always does the measurement.

The measurements are performed in the Hadamard basis: since the qubit created
is always |0> this yields equal probability to get a |+> or |->; however, all
subsequent measurements have 100% probability of returning the same result.

This is an extension of the two-player ping-pong example from:
https://docs.netsquid.org/latest-release/tutorial.components.html
"""

import random
import pydynaa
import netsquid as ns
from netsquid.components import Channel
from netsquid.components.component import Port
from netsquid.components.qchannel import QuantumChannel
from netsquid.components.qmemory import QuantumMemory
from netsquid.components.models.delaymodels import FibreDelayModel


class RingNode(pydynaa.Entity):
    length = 2e-3  # channel length [km]
    counter = 0
    evtype_newqubit = pydynaa.EventType("Create a new qubit and send it out", "")

    def __init__(self):
        self.name = f"RingNode#{self.counter}"
        RingNode.counter += 1

        # Set to True only on the first node sending the qubit
        self.am_i_last = False

        # Create a memory and a quantum channel
        self.qmemory = QuantumMemory("NodeMemory", num_positions=1)
        self.qchannel = QuantumChannel(
            "NodeChannel", length=self.length, models={"delay_model": FibreDelayModel()}
        )
        # Link output from qmemory (pop) to input of channel
        self.qmemory.ports["qout"].connect(self.qchannel.ports["send"])
        # Setup callback function to handle input on quantum memory port "qin0"
        self._wait(
            pydynaa.EventHandler(self._handle_input_qubit),
            entity=self.qmemory.ports["qin0"],
            event_type=Port.evtype_input,
        )
        self.qmemory.ports["qin0"].notify_all_input = True
        # Wait for a new-qubit event
        self._wait(
            pydynaa.EventHandler(self._start),
            entity=self,
            event_type=RingNode.evtype_newqubit,
        )

    def _start(self, event):
        print(f"{ns.sim_time():.1f}: {self.name} qubit transmitted")

        # Send out the qubit from this node
        self.am_i_last = True
        (qubit,) = ns.qubits.create_qubits(1)
        self.qchannel.send(qubit)

    def schedule_qubits(self, event_times):
        # Schedule the events passed
        for e in event_times:
            self._schedule_at(e, self.evtype_newqubit)

    def connect_to_next_node(self, other_entity):
        # Setup this entity to pass incoming qubits to its quantum memory
        self.qmemory.ports["qin0"].connect(other_entity.qchannel.ports["recv"])

    def _handle_input_qubit(self, event):
        # Called upon receiving a qubit from the channel, which is
        # automatically stored in the quantum memory
        if self.am_i_last or random.random() > 0.5:
            [m], [prob] = self.qmemory.measure(positions=[0], observable=ns.X)
            labels_x = ("|+>", "|->")
            print(
                f"{ns.sim_time():.1f}: {self.name} qubit measured "
                f"{labels_x[m]} with probability {prob:.2f}"
            )
        else:
            print(f"{ns.sim_time():.1f}: {self.name} qubit not measured")

        if not self.am_i_last:
            # Do not forward if the transmission started from this node
            self.qmemory.pop(positions=[0])
        else:
            self.am_i_last = False


# Configuration
seed = 42
num_nodes = 4
num_rounds = 10
round_duration = 1000

# Create entities and register them to each other
ns.sim_reset()
nodes = []
for i in range(num_nodes):
    nodes.append(RingNode())

for i in range(num_nodes):
    next_node = i + 1 if (i + 1) < num_nodes else 0
    nodes[i].connect_to_next_node(nodes[next_node])

random.seed(seed)
for i in range(num_rounds):
    nodes[random.randint(0, num_nodes - 1)].schedule_qubits([i * round_duration])

ns.set_random_state(seed=seed)
stats = ns.sim_run()
print(stats)
