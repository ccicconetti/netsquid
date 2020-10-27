"""This module specifies classes that model an application traffic pattern.
"""

import random

__all__ = [
    "Application",
    "SingleConstantApplication",
    "SingleRandomApplication",
    "MultiConstantApplication",
    "MultiRandomApplication",
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
        """Return the list of (A, B, max) tuples for a given timeslot.

        Parameters
        ----------
        timeslot : int
            The timeslot for which we request the information

        Returns
        -------
        list of three-element tuples
            The first two elemente are the two end-points that wish to
            establish an end-to-end entanglement; the third element is the
            maximum number of qubits required by the application.
        
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

class MultiApplication(Application):
    """Abstract class to be used by applications returning multiple pairs.

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
        return self._get_pairs()

    def _get_pairs(self):
        raise NotImplementedError("Class MultiApplication should be inherited")

class SingleConstantApplication(SingleApplication):
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

class SingleRandomApplication(SingleApplication):
    """Return a random pair from a set, all with the same maximum number of qubits.

    The `timeslot` parameter in `get_pairs` is ignored, hence multiple calls
    to method with the same value of `timeslot` will result, in general,
    in a different result.

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

class MultiConstantApplication(MultiApplication):
    """Return always the same pairs, all with the same maximum number of qubits.

    Parameters
    ----------
    name : str
        A name to identify this application.
    pairs : list
        The list of pairs to be returned. Must be non-empty.
    max_qubits : int
        The maximum number of qubits required.
    
    """

    def __init__(self, name, pairs, max_qubits):
        super().__init__(name, max_qubits)

        if not pairs:
            raise ValueError('Cannot initialize MultiConstantApplication with an empty list of pairs')
        self._pairs = []
        for pair in pairs:
            if pair[0] == pair[1]:
                raise ValueError(f'The two end-points cannot be the same: {pair[0]}')
            self._pairs.append([pair[0], pair[1], max_qubits])

    def _get_pairs(self):
        return self._pairs

class MultiRandomApplication(MultiConstantApplication):
    """Return a random list of pairs from a set.
    
    All pairs returned have thxe same maximum number of qubits.

    The `timeslot` parameter in `get_pairs` is ignored, hence multiple calls
    to method with the same value of `timeslot` will result, in general,
    in a different result.

    Parameters
    ----------
    name : str
        A name to identify this application.
    pairs : list
        The list of pairs to be returned. Must be non-empty.
    cardinality : int
        How many pairs to return. Must be smaller than or equal to the number
        of pairs passed as argument to the ctor.
    max_qubits : int
        The maximum number of qubits required.
    
    """

    def __init__(self, name, pairs, cardinality, max_qubits):
        super().__init__(name, pairs, max_qubits)
        
        if cardinality > len(pairs):
            raise ValueError((f'In MultiRandomApplication cardinality is too '
                              f'high ({cardinality}) compared to the number of '
                              f'pairs available ({len(pairs)})'))

        self._cardinality = cardinality

    def _get_pairs(self):
        return random.sample(self._pairs, self._cardinality)
