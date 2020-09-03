"""Simple example of a DepolarNoiseModel in Netsquid

With the given seed, a Bell state

         _           _
   1    |             |
------- | |00> + |11> |
sqrt(2) |_           _|

becomes:

         _           _
   1    |             |
------- | |00> - |11> |
sqrt(2) |_           _|

"""

import logging
import netsquid as ns
from utils import print_dm_single
from netsquid.components.models.qerrormodels import DepolarNoiseModel
from netsquid.qubits import fidelity
from netsquid.qubits import ketstates

# configuration
logging.basicConfig(level=logging.INFO)
ns.set_random_state(seed=1)

# create a Bell pair
q0, q1 = ns.qubits.create_qubits(2, system_name="q")
ns.qubits.operate(q0, ns.H)
ns.qubits.operate([q0, q1], ns.CX)

print_dm_single([q0, q1])
print(f'fidelity = {fidelity([q0, q1], ketstates.b00):.2f}')

# create a depolarizing channel with certainty (probability = 1) to corrupt 
model = DepolarNoiseModel(depolar_rate=1, time_independent=True)

model.compute_model([q0, q1])
print_dm_single([q0, q1])
print(f'fidelity = {fidelity([q0, q1], ketstates.b00):.2f}')
