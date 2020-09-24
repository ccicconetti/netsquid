"""
Example showing how to use Protocols in NetSquid.

There are N nodes connected in a line.

 +----------------------------+     +--------------------------+
 |                            |     |                          |
 |       +------------+       |     |                          |
 |   +--->     LR     +---+   |     |                          |
A|   |   +------------+   v   |B    |port_to_left_channel      |
 +---+                    +---------+                          +------+
 |   ^   +------------+   |   |     |     port_to_right_channel|
 |   +---+     RL     <---+   |     |                          |
 |       +------------+       |     |                          |
 |                            |     |                          |
 +----------------------------+     +--------------------------+
         DirectChannel                         Node

The first node on the left sends a qubit initialized to |0>, which is
then sent to the node on its right. When it reaches the rightmost node, it
gets back towards the source. Each node may apply an X gate to the qubit
before sending it out, with a given probability. The leftmost node performs a
measurement, then sends a new qubit.

This is a modification of the "The ping pong example using protocols" example
from the NetSquid documentation:
https://docs.netsquid.org/latest-release/tutorial.components.html
"""

import random
import netsquid as ns
from netsquid.protocols import NodeProtocol
from netsquid.components import QuantumChannel
from netsquid.nodes import Node, DirectConnection
from netsquid.qubits import qubitapi as qapi

class OriginNodeProtocol(NodeProtocol):
    def __init__(self, node=None, name=None):
        super().__init__(node, name)
        self.measured_bit = 0.
        self.how_many = 0

    def run(self):
        print(f"{ns.sim_time():.1f}: OriginNodeProcol started on {self.node.name}")
        port = self.node.ports["port_to_right_channel"]
        while True:
            # Create a qubit and send it out to the node on the right
            qubit, = qapi.create_qubits(1)
            port.tx_output(qubit)

            # Wait for the qubit to be received back
            yield self.await_port_input(port)
            rx_items = port.rx_input().items
            assert len(rx_items) == 1
            qubit = rx_items[0]
            m, prob = qapi.measure(qubit, ns.Z, discard=True)
            self.how_many += 1
            self.measured_bit += m
            labels_z =  ("|0>", "|1>")
            print(f"{ns.sim_time():.1f}: {self.node.name} measured "
                  f"{labels_z[m]} with probability {prob:.2f}")

class OtherNodeProtocol(NodeProtocol):
    flip_probability = 0.2

    def __init__(self, node, in_port_name, out_port_name):
        super().__init__(node)
        self.in_port = self.node.ports[in_port_name]
        self.out_port = self.node.ports[out_port_name]

    def run(self):
        print(f"{ns.sim_time():.1f}: OtherNodeProtocol started on {self.node.name}")
        while True:
            yield self.await_port_input(self.in_port)
            rx_items = self.in_port.rx_input().items
            assert len(rx_items) == 1
            qubit = rx_items[0]
            if random.random() < self.flip_probability:
                ns.qubits.operate(qubit, ns.X)
            self.out_port.tx_output(qubit)  # send qubit to Ping


# Configuration
sim_duration = 5000 # ns
seed = 42
num_nodes = 4
ns.sim_reset()
random.seed(seed)

# Create nodes
nodes = []
for i in range(num_nodes):
    nodes.append(Node(f"Node#{i}",
                      port_names=["port_to_left_channel", "port_to_right_channel"]))

# Create connections
connections = []
for i in range(1, num_nodes, 1):
    connection = DirectConnection(
        f"Connection#{i}",
        QuantumChannel(f"Channel_LR#{i}", delay=10),
        QuantumChannel(f"Channel_RL#{i}", delay=20)
    )
    nodes[i-1].ports["port_to_right_channel"].connect(connection.ports["A"])
    nodes[i].ports["port_to_left_channel"].connect(connection.ports["B"])
    connections.append(connection)
#
# Create protocols
#
protocols = []

# In the origin node port_to_left_channel is unused
protocols.append(OriginNodeProtocol(nodes[0]))

# In all the middle nodes we create two instances of the protocol: one for
# left-to-right and another for right-to-left transers
for i in range(1, num_nodes-1, 1):
    protocols.append(OtherNodeProtocol(
        nodes[i], "port_to_right_channel", "port_to_left_channel"))
    protocols.append(OtherNodeProtocol(
        nodes[i], "port_to_left_channel", "port_to_right_channel"))

# In the destination node port_to_right_channel is unused
protocols.append(OtherNodeProtocol(
    nodes[num_nodes-1], "port_to_left_channel", "port_to_left_channel"))

# Start all protocols
for protocol in protocols:
    protocol.start()

# Run simulator
ns.set_random_state(seed=seed)
stats = ns.sim_run(duration=sim_duration)
# print(stats)

print(f'measured {protocols[0].measured_bit / protocols[0].how_many} '
      f'(over {protocols[0].how_many} measurements')