import netsquid as ns
import utils

qubits = ns.qubits.create_qubits(2, system_name="q")
utils.print_dm_all(qubits)
utils.measure_all(qubits)
utils.measure_all(qubits, base='X')