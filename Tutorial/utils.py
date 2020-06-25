"""Simple utilities and wrappers"""

import netsquid as ns

def print_dm_all(qubits):
    for qubit in qubits:
        print('{}\n{}\n'.format(qubit.name, ns.qubits.reduced_dm(qubit)))

def measure_all(qubits, base = 'Z'):
    assert base == 'Z' or base == 'X'
    for qubit in qubits:
        if base == 'Z':
            res, prob = ns.qubits.measure(qubit)
            value = '|0>' if res == 0 else '|1>'
        elif base == 'X':
            res, prob = ns.qubits.measure(qubit, observable=ns.X)
            value = '|+>' if res == 0 else '|->'
        print(f'{qubit.name}: {value} (prob. {prob:.2f})')