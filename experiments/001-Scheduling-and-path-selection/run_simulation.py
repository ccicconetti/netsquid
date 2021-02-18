"""
Example experiment with quantum repeaters using uiiit modules.
"""

import os
import random
import logging
import time

import netsquid as ns
from netsquid.protocols import LocalProtocol
from netsquid.components.models.qerrormodels import FibreLossModel

from uiiit.oracle import Oracle
from uiiit.qnetwork import QNetwork
from uiiit.qrepeater import QRepeater
from uiiit.simstat import Conf, Stat
from uiiit.swapprotocol import SwapProtocol
from uiiit.topology import Topology, TopographyDist, Topography2D
from uiiit.traffic import (
    SingleConstantApplication,
    SingleRandomApplication,
    MultiConstantApplication,
    MultiRandomApplication,
    MultiPoissonApplication,
)

__all__ = ['run_simulation']

def run_simulation(conf):
    try:
        logging.info(f'running: {conf}')

        ns.set_qstate_formalism(ns.QFormalism.DM)
        ns.sim_reset()
        random.seed(conf['seed'])
        ns.set_random_state(seed=conf['seed'])

        # Create statistics object that will be returned by the function
        stat = Stat(conf)

        # Configure the topology and topography objects that define the
        # logican and physical layout of nodes, respectively
        if conf['topology'] in ['chain', 'ring', 'grid']:
            topology = Topology(conf['topology'], size=conf['num_nodes'])
            topography = TopographyDist.make_from_topology(
                topology,
                conf['node_distance'] * (1 - conf['node_distance_delta']),
                conf['node_distance'] * (1 + conf['node_distance_delta']))
        else:
            assert conf['topology'] in ['random']
            topology = None
            num_tries = 10000
            for _ in range(num_tries):
                topography = Topography2D(
                    'disc', nodes=conf['num_nodes'],
                    size=conf['size'], threshold=conf['threshold'])
                if len(topography.orphans()) > 0:
                    continue
                topology = Topology('edges', edges=topography.edges())
                if topology.connected():
                    break

            if topology is None or not topology.connected():
                raise RuntimeError(
                    f'Could not find a connected topology after {num_tries} tries')
            if os.path.isdir('topo'):
                mangle = (f"num_nodes={conf['num_nodes']}.size={conf['size']}."
                          f"threshold={conf['threshold']}.seed={conf['seed']}")
                topography.export(f"topo/{mangle}-nodes.dat",
                                f"topo/{mangle}-edges.dat")

        # Set the weights of the topology equal to the physical distance
        # between the nodes sharing that edge
        topography.update_topology(topology)

        stat.count('degree-max', topology.max_degree())
        stat.count('degree-min', topology.min_degree())
        stat.count('degree-avg', topology.avg_degree())

        # Compute the minimum timeslot duration as the sum of the time to
        # transmit a message between the two farthest nodes (using shortest-path)
        # and the time to perform swapping on every node plus corrections on
        # the end point.
        gate_duration = 10 # ns
        est_runtime = topology.diameter() * (5e3 + 2 * gate_duration)
        # Multiply the minimum slot duration assuming that the link quality
        # information had to be conveyed to the Oracle, then memory position
        # info had to be sent back from the Oracle to all nodes.
        #
        # This is a rather conservative worst case assumption.
        est_runtime *= 3
        stat.count('tsduration', est_runtime)  # ns
        logging.debug(f"estimated maximum end-to-end delay = {est_runtime} ns")
        # est_runtime = max(1e6, est_runtime) # use 1 ms for easier inspection

        network_factory = QNetwork(
            source_frequency=1e9 / est_runtime,
            qerr_model=FibreLossModel(p_loss_init=conf['p_loss_init'],
                                    p_loss_length=conf['p_loss_length']))

        qrepeater_factory = QRepeater(dephase_rate=conf['dephase_rate'],
                                    depol_rate=conf['depol_rate'],
                                    gate_duration=conf['gate_duration'])

        network = network_factory.make_network(
            name="QNetwork",
            qrepeater_factory=qrepeater_factory,
            topology=topology,
            topography=topography)

        # List of nodes, sorted in lexycographic order by their names
        node_names = sorted(network.nodes.keys())
        node_names_dict = dict()
        for i in range(len(node_names)):
            node_names_dict[i] = node_names[i]
        topology.assign_names(node_names_dict)
        nodes = [network.nodes[name] for name in node_names]

        protocol = LocalProtocol(nodes=network.nodes)

        # Create the application that will select the pairs of nodes wishing
        # to share entanglement
        app = None
        if 'const' in conf['app']:
            if 'farthest' in conf['app']:
                alice, bob = random.sample(topology.farthest_nodes(), k=1)[0]
                alice_name = topology.get_name_by_id(alice)
                bob_name = topology.get_name_by_id(bob)
                app = SingleConstantApplication(
                    "SingleConstApp", alice_name, bob_name, 0)
        elif 'random' in conf['app'] or 'poisson' in conf['app']:
            if 'single' in conf['app']:
                app = SingleRandomApplication("SingleRandomApp", topology.node_names, 0)
            elif 'multi' in conf['app']:
                pairs = []
                if 'all' in conf['app']:
                    for u in range(topology.num_nodes):
                        for v in range(u):
                            pairs.append([topology.get_name_by_id(u),
                                          topology.get_name_by_id(v)])
                elif 'farthest' in conf['app']:
                    topology_logical = Topology("edges", edges=topology.edges())
                    topology_logical.copy_names(topology)
                    for u, v in topology_logical.farthest_nodes():
                        pair = sorted([topology.get_name_by_id(u),
                                       topology.get_name_by_id(v)])
                        if pair not in pairs:
                            pairs.append(pair)
                if pairs:
                    logging.debug(f'possible random pairs: {pairs}')
                    if 'random' in conf['app']:
                        app = MultiRandomApplication(
                            "MultiRandomApp", pairs, conf['cardinality'], 1)
                    elif 'poisson' in conf['app']:
                        app = MultiPoissonApplication(
                            "MultiPoissonApplication", pairs,
                            conf['cardinality'], 1, conf['seed'])

        if app is None:
            raise ValueError(f"Invalid value for 'app' in configuration: {conf['app']}")

        # Create the oracle and add it to the local protocol
        skip_policy = 'none'
        if 'random-skip' in conf['algorithm']:
            skip_policy = 'random-skip'
        elif 'always-skip' in conf['algorithm']:
            skip_policy = 'always-skip'
        oracle = Oracle(
            'spf' if 'spf' in conf['algorithm'] else 'minmax',
            'hops' if 'hops' in conf['algorithm'] else 'dist',
            skip_policy,
            network, topology, app, stat, est_runtime * 10)
        protocol.add_subprotocol(oracle)

        # Add SwapProtocol to all repeater nodes. Note: we use unique names,
        # since the subprotocols would otherwise overwrite each other in the main protocol.
        for node in nodes:
            subprotocol = SwapProtocol(name=f"Swap_{node.name}", node=node, oracle=oracle)
            protocol.add_subprotocol(subprotocol)
        protocol.start()

        # Start simulation and measure how long it takes, without including the
        # time to setup all the data structures above and any post-processing below.
        sim_start_time = time.monotonic()
        ns.sim_run(est_runtime * conf['timeslots'])
        stat.add('sim_time', time.monotonic() - sim_start_time)

        # Return the statistics collected
        return stat

    except AssertionError:
        logging.error(f"Assertion failed [{e}] in {conf}")
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            raise

    except Exception as e:
        logging.error(f"Unhandled exception [{type(e).__name__}: {e}] in {conf}")

    return None
