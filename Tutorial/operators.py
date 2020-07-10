"""Example inspired from the NetSquid tutorial
on Qubits and quantum computation"""

import logging
from math import exp

import netsquid as ns
from utils import print_dm, print_dm_single

def single_run(seed, apply_noise):
    ns.set_random_state(seed=seed)

    # we create three qubits with the goal of teleporting
    # the first one into the third one

    q = ns.qubits.create_qubits(3, system_name='q', no_state=False)

    # apply noise (if needed)
    if apply_noise:
        prob = ns.qubits.delay_depolarize(q[2], depolar_rate=1e7, delay=20)
        logging.info(f'depolarized with probability {prob:.2f}')

    # first qubit in state
    #  |0> + i|1>
    # ------------
    #   sqrt(2)
    ns.qubits.operate(q[0], ns.H)
    ns.qubits.operate(q[0], ns.S)
    print_dm(q[0])
    to_be_teleported_dm = ns.qubits.reduced_dm(q[0])

    # second and third qubits in Bell state
    ns.qubits.operate(q[1], ns.H)
    ns.qubits.operate([q[1], q[2]], ns.CNOT)
    print_dm_single([q[1], q[2]])

    # Bell state measurement on first two qubits
    ns.qubits.operate([q[0], q[1]], ns.CNOT)
    ns.qubits.operate(q[0], ns.H)

    # measure the first and second first qbits
    m1, prob = ns.qubits.measure(q[0])
    logging.info(f"Measured {labels_z[m1]} with prob {prob:.2f}")
    m2, prob = ns.qubits.measure(q[1])
    logging.info(f"Measured {labels_z[m2]} with prob {prob:.2f}")

    # apply modifiers to the third qubit (the one teleported)
    if m2 == 1:
        ns.qubits.operate(q[2], ns.X)
    if m1 == 1:
        ns.qubits.operate(q[2], ns.Z)
    print_dm(q[2])
    teleported_dm = ns.qubits.reduced_dm(q[2])

    if apply_noise:
        # check fidelity compared to the expected state
        fidelity = ns.qubits.fidelity([q[2]], reference_state=ns.y0, squared=True)
        logging.info(f"Fidelity is {fidelity:.3f}")
        return fidelity

    else:
        # Check exact comparison
        if to_be_teleported_dm.all() == teleported_dm.all():
            logging.info('teleport was successful')
            return 1
        else:
            logging.info('teleport has failed')
            return 0

labels_z = ("|0>", "|1>")
apply_noise = True
num_runs = 1000
logging.basicConfig(level=logging.WARNING)
depol_rate = 1e7
depol_delay = 20 # ns
expected_depol_prob = 1 - exp(-depol_delay * 1e-9 * depol_rate )

sum = 0
for run in range(num_runs):
    sum += single_run(run, apply_noise)

print('avg fidelity over {} runs is {:.2f}, depolarization prob was {:.2f}'.format(
    num_runs,
    sum / num_runs,
    expected_depol_prob))