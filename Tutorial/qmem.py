"""Simple example showing how to work manually with quantum memories
in Netsquid, also with depolarizing channels"""

import netsquid as ns
from netsquid.components import QuantumMemory
from netsquid.components.models.qerrormodels import DepolarNoiseModel
from netsquid.qubits.qubitapi import create_qubits
import netsquid.qubits.operators as ops

from pydynaa import EventType, EventHandler, SimulationEngine, Entity

from utils import print_dm


def mem_used_positions(qmem):
    """Return the list of used (True) or empty (False) positions
    in a quantum memory"""

    ret = []
    for i in range(len(qmem.mem_positions)):
        ret.append(qmem.get_position_used(i))
    return ret


def handlerMeasure(event):
    global good_probability
    print(f"{SimulationEngine().current_time} {event.source}")
    res0 = qmem.measure(positions=0, observable=ops.Z)
    res1 = qmem.measure(positions=1, observable=ops.Z)
    good_probability += 1 if res0[0] == res1[0] else 0


depolar_rate = 1e6  # 1 MHz
num_positions = 2
num_runs = 1000
experiment_duration = 1e3  # 1 us
good_probability = 0.0

for seed in range(num_runs):
    # initialize simulation
    ns.util.simtools.sim_reset()
    ns.set_random_state(seed=seed)

    # create a quantum memory
    qmem = QuantumMemory(name="DepolarMemory", num_positions=num_positions)
    assert len(qmem.mem_positions) == num_positions

    # create depolarizing noise model
    depolar_noise = DepolarNoiseModel(depolar_rate=depolar_rate, time_independent=False)

    # configure the quantum memory positions so that they use the model
    for mem_pos in qmem.mem_positions:
        mem_pos.models["noise_model"] = depolar_noise

    print(f"positions used: {mem_used_positions(qmem)}")

    # create qubits and push them into the quantum memories, one per position
    qubits = create_qubits(num_positions, system_name="a")
    qmem.put(qubits)

    # remove the qubits from the quantum memory
    print(qmem.peek(list(range(num_positions))))
    for i in range(num_positions):
        print(f"positions used: {mem_used_positions(qmem)}")
        print(f"popping qubit in position {i}: {qmem.pop(positions=i)}")

    assert qmem.peek(list(range(num_positions))) == [None] * num_positions
    print(f"positions used: {mem_used_positions(qmem)}")

    # create two qubits and push them both into the same position
    other_qubits = create_qubits(3, system_name="b")
    qmem.put(other_qubits[0], positions=1)
    print(qmem.peek(positions=1))
    qmem.put(other_qubits[1], positions=1)
    print(qmem.peek(positions=1))

    # create a Bell pair in memory using operators (not gates!)
    qmem.put(other_qubits[2], positions=0)
    print(f"positions used: {mem_used_positions(qmem)}")
    qmem.operate(ops.H, positions=[0])
    qmem.operate(ops.CX, positions=[0, 1])

    #
    # schedule a measurement to happen in the future by using pyDynAA
    #
    evtype_meas = EventType("Measurement", "")
    handler_meas = EventHandler(handlerMeasure)
    entity_master = Entity("Master")
    entity_meas = Entity("Measuring box")
    entity_meas._wait(handler_meas, entity=entity_master, event_type=evtype_meas)

    entity_master._schedule_at(experiment_duration, evtype_meas)

    # Run simulation
    run_stats = ns.sim_run()

    # Print statistics
    # print(run_stats)

print(
    (
        f"all went OK with probability {good_probability/num_runs} "
        f"over {num_positions} replications"
    )
)
