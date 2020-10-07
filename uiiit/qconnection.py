"""This module specifies classes that help creation of quantum connections.
"""

import logging

import netsquid as ns
import netsquid.qubits.ketstates as ks
from netsquid.nodes import Connection
from netsquid.protocols.protocol import Protocol
from netsquid.components import Clock, Component
from netsquid.components.qsource import QSource, SourceStatus
from netsquid.qubits.state_sampler import StateSampler
from netsquid.components.cchannel import ClassicalChannel
from netsquid.components.qchannel import QuantumChannel
from netsquid.components.models.delaymodels import FibreDelayModel, FixedDelayModel

__all__ = [
    "ClassicalConnection",
    "EntanglingConnection",
    ]

class PassThroughProtocol(Protocol):
    """A pass-through protocol between two ports.
    
    A protocol that forwards any incoming message from a source port to
    a destination port, also printing the content if debug-level logging is
    enabled

    Parameters
    ----------
    name : str
        Name of the protocol.
    in_port : :class:`netsquid.components.component.Port`
        Input port.
    out_port : :class:`netsquid.components.component.Port`
        Output port.
    """
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


class ClassicalConnection(Connection):
    """A connection that transmits classical messages in one direction, from A to B.

    Copied from netsquid.examples.teleportation.ClassicalConnection.

    Parameters
    ----------
    name : str
       Name of this connection.
    length : float
        End to end length of the connection [km].
    """

    def __init__(self, name, length):
        super().__init__(name=name)
        self.add_subcomponent(ClassicalChannel("Channel_A2B", length=length,
                                               models={"delay_model": FibreDelayModel()}),
                              forward_input=[("A", "send")],
                              forward_output=[("B", "recv")])

class EntanglingConnection(Connection):
    """A connection that generates entanglement.

    Consists of a midpoint holding a quantum source that connects to
    outgoing quantum channels.

    Reused from netsquid.examples.teleportation.EntanglingConnection.

    Parameters
    ----------
    name : str
        Name of the connection.
    length : float
        End to end length of the connection [km].
    source_frequency : float
        Frequency with which midpoint entanglement source generates entanglement [Hz].

    """

    def __init__(self, name, length, source_frequency):
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
        qsource.ports["qout0"].connect(pass_through_a.ports["in_port"])
        pass_through_a.ports["out_port"].connect(qchannel_c2a.ports["send"])
        qsource.ports["qout1"].connect(pass_through_b.ports["in_port"])
        pass_through_b.ports["out_port"].connect(qchannel_c2b.ports["send"])