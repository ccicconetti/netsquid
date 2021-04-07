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

        self.period = period
        self.delay = delay
        self.entangled_qubits = None
        self.terminating = False

    def wait_for_alice(self, alice):
        """Wait for Alice to send an terminating event"""

        self._terminating_handler = pydynaa.EventHandler(self._handle_terminating)
        self._wait(
            self._terminating_handler, entity=alice, event_type=Alice.terminating_evtype
        )

    def _handle_terminating(self, event):
        """Handle terminating event from Alice"""

        logging.info(f"{ns.sim_time():.1f} Charlie received termination event")
        self.terminating = True

    def _entangle_qubits(self, event):
        """Entangle a Bell pair and schedule both the generation of an
        even towards Alice and Bob, and the generation of a fresh Bell pair"""

        if self.terminating:
            return

        self.entangled_qubits = make_bell_pair()
        rand_delay = np.random.exponential(self.delay)
        logging.info(
            f"{ns.sim_time():.1f} Charlie entangled qubits started, "
            f"will require {rand_delay:.2f}"
        )
        self._schedule_after(rand_delay, Charlie.ready_evtype)
        self._schedule_after(self.period, Charlie._generate_evtype)

    def start(self, bob):
        """Trigger start of the simulation"""

        charlie_generate_evexpr = pydynaa.EventExpression(
            source=self, event_type=Charlie._generate_evtype
        )
        bob_ready_evexpr = pydynaa.EventExpression(
            source=bob, event_type=Bob.decoded_evtype
        )
        both_ready_evexpr = charlie_generate_evexpr & bob_ready_evexpr
        self._generate_handler = pydynaa.ExpressionHandler(self._entangle_qubits)
        self._wait(self._generate_handler, expression=both_ready_evexpr)

        self._schedule_now(Charlie._generate_evtype)
        bob.mark_ready()


class Alice(pydynaa.Entity):
    ready_evtype = pydynaa.EventType("DELIVERY_READY", "Qubit ready for delivery.")
    _encode_evtype = pydynaa.EventType("ENCODE", "Encode the payload in the qubit.")
    terminating_evtype = pydynaa.EventType("TERMINATING", "All payloads sent.")

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
        self._wait(self._encode_handler, entity=self, event_type=Alice._encode_evtype)

    def wait_for_charlie(self, charlie):
        """Wait for Charlie to send an entangled qubit"""

        self._qubit_handler = pydynaa.EventHandler(self._handle_qubit)
        self._wait(self._qubit_handler, entity=charlie, event_type=Charlie.ready_evtype)

    def _handle_qubit(self, event):
        """Handle arrival of the entangled qubit from Charlie"""

        self.qubit = event.source.entangled_qubits[0]

        rand_delay = np.random.exponential(self.delay)
        logging.info(
            f"{ns.sim_time():.1f} Alice qubit encoding started, "
            f"will require {rand_delay:.2f}"
        )
        self._schedule_after(rand_delay, Alice._encode_evtype)

    def _handle_encode(self, event):
        """Handle the end of the operations on the qubit"""

        if not self.payloads:
            logging.info(f"{ns.sim_time():.1f} Alice sent all payloads, terminating")
            self._schedule_now(Alice.terminating_evtype)
            return

        payload = self.payloads.pop(0)
        assert 0 <= payload <= 3
        if payload & 2:
            ns.qubits.operate(self.qubit, ns.X)
        if payload & 1:
            ns.qubits.operate(self.qubit, ns.Z)

        logging.info(f"{ns.sim_time():.1f} Alice qubit encoding finished")
        self._schedule_now(Alice.ready_evtype)


class Bob(pydynaa.Entity):
    _decoding_evtype = pydynaa.EventType(
        "DECODING", "Decoding the payload from the qubit."
    )
    decoded_evtype = pydynaa.EventType("DECODED", "Payload decoded from the qubit.")

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
        self.payloads = []

        self._decoding_handler = pydynaa.EventHandler(self._handle_decoding)
        self._wait(self._decoding_handler, entity=self, event_type=Bob._decoding_evtype)

    def wait_for_qubits(self, alice, charlie):
        """Wait for qubits from both Alice and Charlie"""

        charlie_ready_evexpr = pydynaa.EventExpression(
            source=charlie, event_type=Charlie.ready_evtype
        )
        alice_ready_evexpr = pydynaa.EventExpression(
            source=alice, event_type=Alice.ready_evtype
        )
        both_ready_evexpr = charlie_ready_evexpr & alice_ready_evexpr
        self._qubits_handler = pydynaa.ExpressionHandler(self._handle_qubits)
        self._wait(self._qubits_handler, expression=both_ready_evexpr)

    def _handle_qubits(self, event_expression):
        """Handle qubits from Alice and Charlie"""

        self.q0 = event_expression.second_term.atomic_source.qubit
        self.q1 = event_expression.first_term.atomic_source.entangled_qubits[1]

        rand_delay = np.random.exponential(self.delay)
        logging.info(
            f"{ns.sim_time():.1f} Bob qubit decoding started, "
            f"will require {rand_delay:.2f}"
        )
        self._schedule_after(rand_delay, Bob._decoding_evtype)

    def _handle_decoding(self, event_expression):
        """Decode the qubits"""

        ns.qubits.operate([self.q0, self.q1], ns.CX)
        ns.qubits.operate(self.q0, ns.H)
        res0, prob0 = ns.qubits.measure(self.q0, observable=ns.Z)
        res1, prob1 = ns.qubits.measure(self.q1, observable=ns.Z)

        payload = res0 + 2 * res1

        logging.info(
            f"{ns.sim_time():.1f} Bob decoded payload as {payload} "
            f"(with probability {prob0:.2f} {prob1:.2f})"
        )

        self.payloads.append(payload)

        self.mark_ready()

    def mark_ready(self):
        """Generate immediately a DECODED event to indicate that Bob is ready to
        process another qubit"""

        self._schedule_now(Bob.decoded_evtype)


# configuration
seed = 42
logging.basicConfig(level=logging.INFO)

# list of payloads expected
rng = np.random.default_rng(seed=seed)
expected_payloads = list(rng.integers(low=0, high=4, size=10))

# initialize
ns.set_random_state(seed=seed)
np.random.seed(seed)
ns.sim_reset()

# create simulation actors
alice = Alice(delay=20, payloads=expected_payloads.copy())
bob = Bob(delay=30)
charlie = Charlie(period=50, delay=10)

# connect the actors and trigger the start of the events
charlie.wait_for_alice(alice)
alice.wait_for_charlie(charlie)
bob.wait_for_qubits(alice, charlie)
charlie.start(bob)

# run simulation
stats = ns.sim_run()

# print statistics
print(stats)

# check results
if len(expected_payloads) != len(bob.payloads):
    print(
        f"Something went horribly wrong with payloads: "
        f"sent {len(expected_payloads)} vs. "
        f"received {len(bob.payloads)}"
    )

N = len(expected_payloads)
sum = 0
for i, a, b in zip(range(N), expected_payloads, bob.payloads):
    logging.info(f"payload#{i}: {a} {b}")
    if a == b:
        sum += 1
print(
    f"successful decoding rate {sum/N} ({N} payloads), " f"duration {ns.sim_time():.1f}"
)
