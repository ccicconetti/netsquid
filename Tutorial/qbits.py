"""Example inspired from the NetSquid tutorial
on Qubits and quantum computation"""

import netsquid as ns
from utils import print_dm

q0, q1 = ns.qubits.create_qubits(2, system_name="q")

print_dm(q0)
ns.qubits.combine_qubits([q0, q1])

print_dm(q0)

res = ns.qubits.measure(q0)
print(f"measurement of {q0.name} through Z: {res}")

res = ns.qubits.measure(q1, observable=ns.X)
print(f"measurement of {q1.name} through X: {res}")
print_dm([q0, q1])

q2, q3 = ns.qubits.create_qubits(2, system_name="q", no_state=True)
try:
    print_dm([q2, q3])
except Exception:
    print(f"{q2.name} and {q3.name} don't have a state")
ns.qubits.assign_qstate([q2, q3], ns.h10)  # assign |-+>
print_dm([q2, q3])
