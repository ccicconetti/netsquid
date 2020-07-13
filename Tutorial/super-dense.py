"""
Example showing super-dense coding transfer of information.

Inspired from NetSquid's Discrete event simulation tutorial at:

https://docs.netsquid.org/latest-release/tutorial.pydynaa.html

Charlie: prepares the qbits in the Bell state and sends them to Alice and Bob.

Alice: encodes two bits (b1 and b2) into the qbit it receives from Charlie,
which then forwards to Bob.

Bob: receives a qbit from Charlie and another from Alice, then it decodes
the two-bit information from it.

Regarding the protocol see, e.g.,:

https://en.wikipedia.org/wiki/Superdense_coding
"""

import logging
import numpy as np

import netsquid as ns
import pydynaa
from utils import make_bell_pair

class Charlie(pydynaa.Entity):
    ready_evtype = pydynaa.EventType("QUBITS_READY", "Entangled qubits are ready.")
    _generate_evtype = pydynaa.EventType("GENERATE", "Generate entangled qubits.")

    def __init__(self, period, delay):
        """
        Parameters
        ----------
        period : float
            The entanglement period, i.e., at which to trigge GENERATE events.
        delay : float
            The average delay required to entangle the qubits, drawn from an
            exponential r.v.
        """

        # Initialise Charlie by entangling qubits after every generation event
        self.period = period
        self.delay = delay
        self.entangled_qubits = None
        self._generate_handler = pydynaa.EventHandler(self._entangle_qubits)
        self._wait(self._generate_handler, entity=self,
                   event_type=Charlie._generate_evtype)

    def _entangle_qubits(self, event):
        # Callback function that entangles qubits and schedules an
        # entanglement ready event
        self.entangled_qubits = make_bell_pair()
        rand_delay = np.random.exponential(self.delay)
        logging.info(f'{ns.sim_time():.1f} Charlie entangled qubits started, '
                     f'will require {rand_delay:.2f}')
        self._schedule_after(rand_delay, Charlie.ready_evtype)
        self._schedule_after(self.period, Charlie._generate_evtype)

    def start(self):
        # Begin generating entanglement
        self._schedule_now(Charlie._generate_evtype)

class Alice(pydynaa.Entity):
    ready_evtype = pydynaa.EventType("DELIVERY_READY", "Qubit ready for delivery.")
    _encode_evtype = pydynaa.EventType("ENCODE", "Encode the payload in the qubit.")

    def __init__(self, delay, payloads):
        """
        Parameters
        ----------
        delay : float
            The average delay required to perform the quantum operations, drawn
            from an exponential r.v.
        payloads: list
            List of payloads (0-3) to send to Bob.
        """

        self.delay = delay
        self.payloads = payloads
        self.qubit = None
        self._encode_handler = pydynaa.EventHandler(self._handle_encode)
        self._wait(self._encode_handler, entity=self,
                   event_type=Alice._encode_evtype)

    def wait_for_charlie(self, charlie):
        # Setup Alice to wait for an entanglement qubit from Charlie
        self._qubit_handler = pydynaa.EventHandler(self._handle_qubit)
        self._wait(self._qubit_handler, entity=charlie,
                   event_type=Charlie.ready_evtype)

    def _handle_qubit(self, event):
        """Handle arrival of entangled qubit"""

        self.qubit = event.source.entangled_qubits[0]

        rand_delay = np.random.exponential(self.delay)
        logging.info(f'{ns.sim_time():.1f} Alice qubit encoding started, '
                     f'will require {rand_delay:.2f}')
        self._schedule_after(rand_delay, Alice._encode_evtype)

    def _handle_encode(self, event):
        """Handle the end of the operations on the qubit"""

        payload = 3 # XXX
        if payload & 2:
            ns.qubits.operate(self.qubit, ns.X)
        if payload & 1:
            ns.qubits.operate(self.qubit, ns.Z)

        logging.info(f'{ns.sim_time():.1f} Alice qubit encoding finished')
        self._schedule_now(Alice.ready_evtype)

class Bob(pydynaa.Entity):
    _decoding_evtype = pydynaa.EventType("DECODING", "Decoding the payload from the qubit.")
  
    def __init__(self, delay):
        """
        Parameters
        ----------
        delay : float
            The average delay required to perform the quantum operations, drawn
            from an exponential r.v.
        """

        self.delay = delay
        self.q0 = None
        self.q1 = None

        self._decoding_handler = pydynaa.EventHandler(self._handle_decoding)
        self._wait(self._decoding_handler, entity=self,
                   event_type=Bob._decoding_evtype)


    def wait_for_qubits(self, alice, charlie):
        """Wait for qubits from both Alice and Charlie"""

        charlie_ready_evexpr = pydynaa.EventExpression(
            source=charlie, event_type=Charlie.ready_evtype)
        alice_ready_evexpr = pydynaa.EventExpression(
            source=alice, event_type=Alice.ready_evtype)
        both_ready_evexpr = charlie_ready_evexpr & alice_ready_evexpr
        self._qubits_handler = pydynaa.ExpressionHandler(self._handle_qubits)
        self._wait(self._qubits_handler, expression=both_ready_evexpr)

    def _handle_qubits(self, event_expression):
        """Handle qubits from Alice and Charlie"""

        self.q0 = event_expression.second_term.atomic_source.qubit
        self.q1 = event_expression.first_term.atomic_source.entangled_qubits[1]

        rand_delay = np.random.exponential(self.delay)
        logging.info(f'{ns.sim_time():.1f} Bob qubit decoding started, '
                     f'will require {rand_delay:.2f}')
        self._schedule_after(rand_delay, Bob._decoding_evtype)

    def _handle_decoding(self, event_expression):
        """Decode the qubits"""

        ns.qubits.operate([self.q0, self.q1], ns.CX)
        ns.qubits.operate(self.q0, ns.H)
        res0, prob0 = ns.qubits.measure(self.q0, observable=ns.Z)
        res1, prob1 = ns.qubits.measure(self.q1, observable=ns.Z)

        payload = res0 + 2 * res1
    
        logging.info(f'{ns.sim_time():.1f} Bob decoded payload as {payload} '
                     f'(with probability {prob0:.2f} {prob1:.2f})')

# configuration
seed = 42
duration = 100000
logging.basicConfig(level=logging.INFO)

# initialize
ns.set_random_state(seed=seed)
np.random.seed(seed)
ns.sim_reset()

# create simulation actors
expected_payloads = range(4)
alice = Alice(delay=20, payloads=expected_payloads)
bob = Bob(delay=30)
charlie = Charlie(period=50000, delay=10)

# start the actors
charlie.start()
alice.wait_for_charlie(charlie)
bob.wait_for_qubits(alice, charlie)

# run simulation
stats = ns.sim_run(end_time=duration)

# print statistics
print(stats)