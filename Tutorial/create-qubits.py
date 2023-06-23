"""Trivial example of initialization and measurements on qubits"""

import netsquid as ns
import utils

qubits = ns.qubits.create_qubits(2, system_name="q")
utils.print_dm_all(qubits)

# create a Bell Pair b_00
ns.qubits.operate(qubits[0], ns.H)
ns.qubits.operate([qubits[0], qubits[1]], ns.CX)

# the first bit can be 0 or 1 in the computational basis
res0 = ns.qubits.measure(qubits[0], observable=ns.Z)
assert 0.49 <= res0[1] <= 0.51

# the second qubit must the same as the first one
res1 = ns.qubits.measure(qubits[1], observable=ns.Z)
assert res0[0] == res1[0]
assert res1[1] == 1

# if we measure again we find the same results as before, with certainty
again_res0 = ns.qubits.measure(qubits[0], observable=ns.Z)
again_res1 = ns.qubits.measure(qubits[1], observable=ns.Z)
assert again_res0[0] == res0[0]
assert again_res1[0] == res0[0]
assert again_res0[1] == 1
assert again_res1[1] == 1
