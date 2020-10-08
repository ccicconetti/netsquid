"""This module specifies classes that model an application traffic pattern.
"""

__all__ = [
    "Application",
    "SinglePairConstantApplication",
    ]

class Application:
    """Generic class for quantum applications.
    """
    def __init__(self):
        pass

    def get_pairs(self, timeslot):
        """Return the list of (A, B, min) tuples for a given timeslot.

        Parameters
        ----------
        timeslot : int
            The timeslot for which we request the information

        Returns
        -------
        list of three-element tuples
            The first two elemente are the two end-points that wish to
            establish an end-to-end entanglement; the third element is the
            mininum number of qubits required by the application.
        
        """
        raise NotImplementedError("Class Application should be inherited")

class SinglePairConstantApplication(Application):
    """Return always the same pair.

    Parameters
    ----------
    alice : str
        The name of the first end-point
    bob : str
        The name of the second end-point
    
    """

    def __init__(self, alice, bob):
        self._alice = alice
        self._bob = bob

    def get_pairs(self, timeslot):
        return (self._alice, self._bob, 1)
