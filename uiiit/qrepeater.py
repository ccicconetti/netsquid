"""This module specifies a class that helps creation of quantum repeaters.
"""

from netsquid.components import QuantumProcessor, PhysicalInstruction
from netsquid.components.models.qerrormodels import DepolarNoiseModel, DephaseNoiseModel
from netsquid.components.instructions import INSTR_MEASURE_BELL, INSTR_X, INSTR_Z

__all__ = ["QRepeater"]


class QRepeater:
    """Factory to create a quantum processor for a quantum repeater.

    It has as many memory positions as connections with others nodes
    for end-to-end entanglement via teleportation.

    Parameters
    ----------
    dephase_rate : float
        Dephase noise rate [Hz].
    depol_rate : float
        Depolarization noise rate [Hz].
    gate_duration : float
        Time required to perform each quantum operation [ns].
    """

    def __init__(self, dephase_rate, depol_rate, gate_duration):
        self._gate_noise_model = DephaseNoiseModel(dephase_rate)
        self._mem_noise_model = DepolarNoiseModel(depol_rate)
        self._physical_instructions = [
            PhysicalInstruction(
                INSTR_X, duration=gate_duration, q_noise_model=self._gate_noise_model
            ),
            PhysicalInstruction(
                INSTR_Z, duration=gate_duration, q_noise_model=self._gate_noise_model
            ),
            PhysicalInstruction(INSTR_MEASURE_BELL, duration=gate_duration),
        ]

    def make_qprocessor(self, name, mem_positions):
        """Create a quantum processor with the class-specified characteristics.

        Parameters
        ----------
        name : str
            Name of the quantum processor.
        mem_positions : int
            Number of quantum memory positions.

        Returns
        -------
        :class:`~netsquid.components.qprocessor.QuantumProcessor`
            A quantum processor to specification.

        """
        return QuantumProcessor(
            name,
            num_positions=mem_positions,
            fallback_to_nonphysical=False,
            mem_noise_models=[self._mem_noise_model] * mem_positions,
            phys_instructions=self._physical_instructions,
        )
