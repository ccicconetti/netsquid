"""
Example showing how two Nodes communicating via classical channels.

This is a modification of the "Nodes and Connections" example from NetSquid:
https://docs.netsquid.org/latest-release/tutorial.components.html
"""

import random
import pydynaa
import netsquid as ns
from netsquid.nodes import Node
from netsquid.nodes.connections import Connection
from netsquid.components import ClassicalChannel
from netsquid.components.models import FibreDelayModel

class DiffusionNode(Node):
    """Node receiving messages and re-transmitting them unless it is
    the final destination"""

    counter = 0

    def __init__(self):
        super().__init__(f"DN#{DiffusionNode.counter}",
                         port_names=["in_port", "out_port"])
        DiffusionNode.counter += 1

        self.ports['in_port'].bind_input_handler(self._handle_message)

    def _handle_message(self, message):
        # Callback function that handles incoming messages
        assert len(message.items) == 2

        if message.items[0] == self.name:
            print(f"{ns.sim_time():.1f}: {self.name} message received on final destination: {message.items[1]}")
        else:
            # for port in self.ports:
            self.ports['out_port'].tx_output(message.items)
            # self.nodes[0].ports['out_port'].tx_output(['DN#1', 'Hello'])
            print(f"{ns.sim_time():.1f}: {self.name} forwarding message: {message.items[1]}")

class ClassicalConnection(Connection):
    """Bi-directional connection with two classical channels"""

    def __init__(self, length):
        super().__init__(name="ClassicalConnection")

        self.add_subcomponent(
            ClassicalChannel(f"Channel_A2B", length=length,
            models={"delay_model": FibreDelayModel()}))

        self.ports['A'].forward_input(
            self.subcomponents["Channel_A2B"].ports['send'])
        self.subcomponents["Channel_A2B"].ports['recv'].forward_output(
            self.ports['B'])

        self.add_subcomponent(
            ClassicalChannel(f"Channel_B2A", length=length,
            models={"delay_model": FibreDelayModel()}))
        self.add_ports(['A2', 'B2'])
        self.ports['A2'].forward_input(
            self.subcomponents["Channel_B2A"].ports['send'])
        self.subcomponents["Channel_B2A"].ports['recv'].forward_output(
            self.ports['B2'])

class MessageGenerator(pydynaa.Entity):
    evtype_newmessage = pydynaa.EventType("New message event", "")

    def __init__(self, nodes, period):
        self.nodes = nodes
        self.period = period
        self._wait(pydynaa.EventHandler(self._send),
            entity=self, event_type=MessageGenerator.evtype_newmessage)
        self._schedule_now(self.evtype_newmessage)

        self.message_counter = 0

    def _send(self, event):
        # Send a message
        src_node = random.choice([0, 1])
        dst_node_name = random.choice(['DN#0', 'DN#1'])
        print(f"{ns.sim_time():.1f}: new message scheduled to be sent "
              f"from DN#{src_node} to {dst_node_name}")

        self.nodes[src_node].ports['out_port'].tx_output(
            [dst_node_name, f'Hello{self.message_counter}'])

        # Scheduler another message in the next period
        self.message_counter += 1
        self._schedule_after(self.period, self.evtype_newmessage)

# Configuration
length = 20e-3 # 200 m
sim_duration = 5000 # ns
seed=42
ns.sim_reset()
random.seed(seed)

# Create nodes
a = DiffusionNode()
b = DiffusionNode()

# Create classical channels
c = ClassicalConnection(length)

# Connect the nodes
a.ports['out_port'].connect(c.ports['A'])
b.ports['in_port'].connect(c.ports['B'])
b.ports['out_port'].connect(c.ports['A2'])
a.ports['in_port'].connect(c.ports['B2'])

# Create an instance of the message generator
msg_generator = MessageGenerator(nodes=[a, b], period=1000)

# Run simulator
ns.set_random_state(seed=seed)
stats = ns.sim_run(duration=sim_duration)
# print(stats)