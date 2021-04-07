"""Simple utilities and wrappers"""

from collections.abc import Iterable

import logging

import netsquid as ns


def print_dm_all(qubits):
    for qubit in qubits:
        print("{}\n{}\n".format(qubit.name, ns.qubits.reduced_dm(qubit)))


def measure_all(qubits, base="Z"):
    assert base == "Z" or base == "X"
    for qubit in qubits:
        if base == "Z":
            res, prob = ns.qubits.measure(qubit)
            value = "|0>" if res == 0 else "|1>"
        elif base == "X":
            res, prob = ns.qubits.measure(qubit, observable=ns.X)
            value = "|+>" if res == 0 else "|->"
        print(f"{qubit.name}: {value} (prob. {prob:.2f})")


def make_bell_pair():
    """Create and return a Bell pair of qubits"""

    q1, q2 = ns.qubits.create_qubits(2)
    ns.qubits.operate(q1, ns.H)
    ns.qubits.operate([q1, q2], ns.CNOT)
    return [q1, q2]


def print_nodes_all(nodes):
    """Print information on a bunch of nodes"""

    for node in nodes:
        print(f"{node.name}, ports {[x[0] for x in node.ports.items()]}")


def print_dm_single(qubit):
    """Print the name and reduced density matrix of a qubit"""
    name = None
    if isinstance(qubit, Iterable):
        name = ",".join([x.name for x in qubit])
    else:
        name = qubit.name
    logging.info(f"{name}, reduce dm:\n{ns.qubits.reduced_dm(qubit)}")


def print_dm(qubits):
    """Print the name and reduced density matrix of one or more qubits"""
    if isinstance(qubits, Iterable):
        for q in qubits:
            print_dm_single(q)
    else:
        print_dm_single(qubits)
