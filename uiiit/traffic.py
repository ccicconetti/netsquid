"""This module specifies classes that model an application traffic pattern.
"""

import random

__all__ = [
    "Application",
    "SinglePairConstantApplication",
    ]

class Application:
    """Generic class for quantum applications.

    Parameters
    ----------
    name : str
        A name to identify this application.

    Properties
    ----------
    name : str
        A name to identify this application.
    """
    def __init__(self, name):
        self.name = name

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

class SingleApplication(Application):
    """Abstract class to be used by applications returning a single pair.

    Also, the number of qubits is always the same, as set in the ctor.

    Parameters
    ----------
    name : str
        A name to identify this application.
    max_qubits : int
        The maximum number of qubits required.

    Raises
    ------
    ValueError
        The maximum number of qubits required is negative.

    """
    def __init__(self, name, max_qubits):
        super().__init__(name)

        if max_qubits < 0:
            raise ValueError("Cannot have negative number of qubits specified")

        self._max_qubits = max_qubits

    def get_pairs(self, timeslot):
        # timeslot is unused
        pair = self._get_single_pair()
        return [(pair[0], pair[1], self._max_qubits)]

    def _get_single_pair(self):
        raise NotImplementedError("Class SingleApplication should be inherited")

class SinglePairConstantApplication(SingleApplication):
    """Return always the same pair, all with the same maximum number of qubits.

    Parameters
    ----------
    name : str
        A name to identify this application.
    alice : str
        The name of the first end-point
    bob : str
        The name of the second end-point
    max_qubits : int
        The maximum number of qubits required.
    
    """

    def __init__(self, name, alice, bob, max_qubits):
        super().__init__(name, max_qubits)

        if alice == bob:
            raise ValueError(f"Cannot use the same name in SinglePairConstantApplication: {alice}")

        self._alice = alice
        self._bob = bob

    def _get_single_pair(self):
        return [self._alice, self._bob]

class SingleRandomPairs(SingleApplication):
    """Return a random pair from a set, all with the same maximum number of qubits.

    Parameters
    ----------
    name : str 
        A name to identify this application.
    node_names : iterable
        The possible names from which to extract the pair
    max_qubits : int
        The maximum number of qubits required.
    
    """

    def __init__(self, name, node_names, max_qubits):
        super().__init__(name, max_qubits)

        self._node_names = set(node_names)

        if len(self._node_names) <= 1:
            raise ValueError(("Invalid cardinality of set of names passed to "
                              f"SingleRandomPairs: {len(self._node_names)}"))

    def _get_single_pair(self):
        return random.sample(self._node_names, 2)
