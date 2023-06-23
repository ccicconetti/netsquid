"""
Example showing how to use signals in Protocols in NetSquid, see:

https://docs.netsquid.org/latest-release/tutorial.protocols.html
"""

import netsquid as ns
from netsquid.protocols import LocalProtocol, Signals
from netsquid.nodes import Node


class SignalProtocol(LocalProtocol):
    time_bomb = 50

    def run(self):
        for x in range(10):
            timer = self.await_timer(10)
            yield timer

            if ns.sim_time() >= self.time_bomb:
                self.stop()

            self.send_signal(Signals.SUCCESS, x)
            print(f"{ns.sim_time():.1f}: {self.name} waiting done, sent {x}")


class AwaitProtocol(LocalProtocol):
    def __init__(self, nodes=None, name=None, signal_protocol=None):
        super().__init__(nodes=nodes, name=name)
        self.add_subprotocol(signal_protocol, "signal_protocol")

    def start(self):
        super().start()
        self.start_subprotocols()

    def run(self):
        while True:
            event = self.await_signal(
                sender=self.subprotocols["signal_protocol"],
                signal_label=Signals.SUCCESS,
            )
            yield event
            res = self.subprotocols["signal_protocol"].get_signal_result(
                Signals.SUCCESS
            )
            print(f"{ns.sim_time():.1f}: {self.name} {res}")


a_node = Node("SignalTx")
b_node = Node("SignalRx1")
c_node = Node("SignalRx2")

abc_nodes = {a_node.name: a_node, b_node.name: b_node, c_node.name: c_node}

a_protocol = SignalProtocol(abc_nodes)
assert a_protocol.is_connected

b_protocol = AwaitProtocol(nodes=abc_nodes, name="Await1", signal_protocol=a_protocol)
assert b_protocol.is_connected

c_protocol = AwaitProtocol(nodes=abc_nodes, name="Await2", signal_protocol=a_protocol)
assert c_protocol.is_connected

print(a_protocol.can_signal_to(b_protocol))  # True
print(a_protocol.can_signal_to(c_protocol))  # True

a_protocol.start()
b_protocol.start()
c_protocol.start()

stats = ns.sim_run()
