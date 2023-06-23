"""
Multi-segment ping pong example.

Nodes are connected in a line, e.g. with four player:

  -->   -->   -->
A     B     C     D
  <--   <--   <--

A prepares a qubit and sends it to B, which sends it to C,
etc. until the qubit gets back to A, which sends it out again for ever
(or until the simulation stops).

Even nodes perform measurements in the Hadamard basis, odd
nodes in the standard basis.

This is an extension of the two-player ping-pong example in
https://docs.netsquid.org/latest-release/quick_start.html
"""

import sys

import netsquid as ns
from netsquid.nodes import Node
from netsquid.components.models import DelayModel
from netsquid.components import QuantumChannel
from netsquid.nodes import DirectConnection
from netsquid.protocols import NodeProtocol
import utils


class PingPongDelayModel(DelayModel):
    def __init__(self, speed_of_light_fraction, standard_deviation):
        super().__init__()
        # (the speed of light is about 300,000 km/s)
        self.properties["speed"] = speed_of_light_fraction * 3e5
        self.properties["std"] = standard_deviation
        self.required_properties = ["length"]  # in km

    def generate_delay(self, **kwargs):
        avg_speed = self.properties["speed"]
        std = self.properties["std"]
        # The 'rng' property contains a random number generator
        # We can use that to generate a random speed
        speed = self.properties["rng"].normal(avg_speed, avg_speed * std)
        delay = 1e9 * kwargs["length"] / speed  # in nanoseconds
        return delay


class PingPongProtocol(NodeProtocol):
    def __init__(self, node, observable, rcv_port, dst_port, qubit=None):
        super().__init__(node)
        self.observable = observable
        self.qubit = qubit
        self.rcv_port = rcv_port
        self.dst_port = dst_port
        # Define matching pair of strings for pretty printing of basis states:
        self.basis = ["|0>", "|1>"] if observable == ns.Z else ["|+>", "|->"]

    def run(self):
        if self.qubit is not None:
            # Send (TX) qubit to the other node via port's output:
            self.node.ports[self.dst_port].tx_output(self.qubit)

        while True:
            # Wait (yield) until input has arrived on our port:
            yield self.await_port_input(self.node.ports[self.rcv_port])
            # Receive (RX) qubit on the port's input:
            message = self.node.ports[self.rcv_port].rx_input()
            qubit = message.items[0]
            meas, prob = ns.qubits.measure(qubit, observable=self.observable)
            print(
                f"{ns.sim_time():5.1f}: {self.node.name} measured "
                f"{self.basis[meas]} with probability {prob:.2f}"
            )

            # Send (TX) qubit to the other node via connection:
            self.node.ports[self.dst_port].tx_output(qubit)


sim_duration_ns = 300  # simulation duration, in ns
num_nodes = 7  # number of nodes in the string
distance = 2.74 / 1000  # default unit of length in channels is km
delay_model = PingPongDelayModel(speed_of_light_fraction=0.5, standard_deviation=0.05)

# Create nodes
nodes = []
for i in range(num_nodes):
    nodes.append(Node(name="node{}".format(i)))

# Create channels (fwd = left to right, bck = right to left)
channels_fwd = []
channels_bck = []
assert num_nodes >= 2
for i in range(num_nodes - 1):
    channels_fwd.append(
        QuantumChannel(
            name="conn{}->{}".format(i, i + 1),
            length=distance,
            models={"delay_model": delay_model},
        )
    )
    channels_bck.append(
        QuantumChannel(
            name="conn{}<-{}".format(i + 1, i),
            length=distance,
            models={"delay_model": delay_model},
        )
    )

# Create the connections between nodes
connections = []
for i in range(num_nodes - 1):
    connections.append(
        DirectConnection(
            name="link{}-{}".format(i, i + 1),
            channel_AtoB=channels_fwd[i],
            channel_BtoA=channels_bck[i],
        )
    )

# Connect the nodes to one another (and create ports in the process)
port_names = dict()
for i in range(num_nodes - 1):
    res = nodes[i].connect_to(remote_node=nodes[i + 1], connection=connections[i])
    if i not in port_names:
        port_names[i] = []
    port_names[i].append(res[0])
    if i + 1 not in port_names:
        port_names[i + 1] = []
    port_names[i + 1].append(res[1])

# Prepare the qubit which will be sent back and forth
qubits = ns.qubits.create_qubits(1, system_name="q")
ns.qubits.operate(qubits[0], ns.H)

# Create the protocol instances
protocols_fwd = []
protocols_bck = []
observables = [ns.X, ns.Z]
for i in range(num_nodes - 1):
    # Forward protocol
    rcv_port = port_names[i][0]
    if len(port_names[i]) == 1:
        dst_port = rcv_port
    else:
        dst_port = port_names[i][1]

    qubit_or_none = qubits[0] if i == 0 else None

    protocols_fwd.append(
        PingPongProtocol(
            nodes[i],
            observable=observables[i % 2],
            rcv_port=rcv_port,
            dst_port=dst_port,
            qubit=qubit_or_none,
        )
    )

    # Backward protocol
    dst_port = port_names[i + 1][0]
    if len(port_names[i + 1]) == 1:
        rcv_port = dst_port
    else:
        rcv_port = port_names[i + 1][1]

    protocols_bck.append(
        PingPongProtocol(
            nodes[i + 1],
            observable=observables[(i + 1) % 2],
            rcv_port=rcv_port,
            dst_port=dst_port,
        )
    )

# Start all the protocols
for proto in protocols_fwd + protocols_bck:
    proto.start()

utils.print_nodes_all(nodes)

# Run simulation
run_stats = ns.sim_run(duration=sim_duration_ns)

# Print statistics
print(run_stats)
