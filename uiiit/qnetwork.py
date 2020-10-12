"""This module specifies classes that help creation of quantum networks.
"""

import logging
import numpy as np
import random

from netsquid.nodes import Node, Network, Connection

from uiiit.qconnection import ClassicalConnection, EntanglingConnection

__all__ = [
    "QNetworkUniform"
    ]

class QNetworkUniform:
    """Factory to create a network made of quantum repeaters.

    Parameters
    ----------
    node_distance : float
        Distance between adjacent nodes [km].
    node_distance_error : float
        Add a random distance randomly drawn from -node_distance_error/2 and
        node_distance_error/2 to the actual distance between any two nodes [km].
    source_frequency : float
        Frequency at which sources generate qubits [Hz].
    qerr_model: :class:`netsquid.components.models.qerrormodels.QuantumErrorModel`
        The quantum error model to use.
    """
    def __init__(self, node_distance, node_distance_error, source_frequency, qerr_model):
        self._node_distance = node_distance
        self._node_distance_error = node_distance_error
        self._source_frequency = source_frequency
        self._qerr_model = qerr_model

    def make_network(self, name, qrepeater_factory, topology):
        """Create a quantum network with the class-specified characteristics. 

        Parameters
        ----------
        name : str
            Name of the network.
        qrepeater_factory : :class:`~uiiit.topology.QRepeater`
            Quantum repeater factory.
        topology : :class:`~uiiit.topology.Topology`
            Network topology.

        Returns
        -------
        :class:`~netsquid.nodes.network.Network`
            Network component with all nodes and connections as subcomponents.
        """

        network = Network(name)

        # To prepend leading zeros to the number
        num_zeros = int(np.log10(topology.num_nodes)) + 1

        # Create nodes and add them to the network
        nodes = []
        for i in range(topology.num_nodes):
            nodes.append(Node(
                f"Node_{i:0{num_zeros}d}",
                qmemory=qrepeater_factory.make_qprocessor(
                    f"qproc_{i}", len(topology.neigh(i)))))
        network.add_nodes(nodes)

        # Create quantum and classical connections
        for [u, v] in topology.biedges():
            lhs_node, rhs_node = nodes[u], nodes[v]
            lhs_id, rhs_id = topology.incoming_id(u, v), topology.incoming_id(v, u)

            logging.info((f"creating quantum and classical connections between "
                          f"{lhs_node.name} (port {lhs_id}) and "
                          f"{rhs_node.name} (port {rhs_id})"))

            # Create a bidirectional quantum connection between the two nodes
            # that also emits periodically entangled qubits
            qconn = EntanglingConnection(
                name=f"qconn_{u}-{v}",
                length=self._get_length(),
                source_frequency=self._source_frequency)

            # Add quantum noise model
            for channel_name in ['qchannel_C2A', 'qchannel_C2B']:
                qconn.subcomponents[channel_name].models['quantum_noise_model'] = self._qerr_model

            # Connect the two nodes lhs and rhs via the entangling connection
            network.add_connection(
                lhs_node, rhs_node, connection=qconn, label="quantum",
                port_name_node1=f"qcon{lhs_id}", port_name_node2=f"qcon{rhs_id}")

            # Forward incoming qubits to the quantum memory positions of the nodes
            lhs_node.ports[f"qcon{lhs_id}"].forward_input(lhs_node.qmemory.ports[f"qin{lhs_id}"])
            rhs_node.ports[f"qcon{rhs_id}"].forward_input(rhs_node.qmemory.ports[f"qin{rhs_id}"])

            # Create a classical connection between the two nodes
            cconn = ClassicalConnection(name=f"cconn_{u}-{v}", length=self._node_distance)
            network.add_connection(
                lhs_node, rhs_node, connection=cconn, label="classical",
                port_name_node1=f"ccon{lhs_id}", port_name_node2=f"ccon{rhs_id}")

        return network

    def _get_length(self):
        return self._node_distance +\
            random.uniform(-self._node_distance_error/2,
                           self._node_distance_error/2)