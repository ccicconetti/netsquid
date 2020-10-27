__author__  = "Claudio Cicconetti"
__version__ = "0.1.0"
__license__ = "MIT"

import unittest
from traffic import SinglePairConstantApplication, SingleRandomPairs

class TestApplication(unittest.TestCase):
    def test_single_pair_constant_application(self):
        app = SinglePairConstantApplication("App", "alice", "bob", 1)

        self.assertEqual("App", app.name)

        for timeslot in range(10):
            pairs = app.get_pairs(timeslot)
            self.assertEqual(1, len(pairs))
            self.assertEqual(("alice", "bob", 1), pairs[0])

        app2 = SinglePairConstantApplication("", "alice", "bob", 2)
    
        self.assertEqual([("alice", "bob", 2)], app2.get_pairs(0))

        with self.assertRaises(ValueError):
            SinglePairConstantApplication("", "alice", "alice", 2)

        _ = SinglePairConstantApplication("", "alice", "bob", 0)

        with self.assertRaises(ValueError):
            SinglePairConstantApplication("", "alice", "bob", -1)

    def test_single_random_pairs(self):
        with self.assertRaises(ValueError):
            SingleRandomPairs("App", [], 1)

        with self.assertRaises(ValueError):
            SingleRandomPairs("App", ["alice"], 1)

        _ = SingleRandomPairs("App", ["alice", "bob"], 0)

        with self.assertRaises(ValueError):
            SingleRandomPairs("App", ["alice", "bob"], -1)

        app = SingleRandomPairs("App", ["alice", "bob"], 7)

        self.assertEqual("App", app.name)

        for timeslot in range(10):
            pairs = app.get_pairs(timeslot)
            self.assertEqual(1, len(pairs))
            pair = pairs[0]
            self.assertTrue(pair[0] in ["alice", "bob"])
            if pair[0] == "alice":
                self.assertEqual("bob", pair[1])
            else:
                self.assertEqual("alice", pair[1])
            self.assertEqual(7, pair[2])

        node_names = set([f'{x}' for x in range(10)])

        app2 = SingleRandomPairs("", node_names, 1)

        found_names = set()
        for timeslot in range(100):
            pair = app2.get_pairs(timeslot)[0]
            found_names.add(pair[0])

        self.assertEqual(node_names, found_names)

if __name__ == '__main__':
    unittest.main()