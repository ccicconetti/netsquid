__author__ = "Claudio Cicconetti"
__version__ = "0.1.0"
__license__ = "MIT"

import unittest
from traffic import (
    SingleConstantApplication,
    SingleRandomApplication,
    MultiConstantApplication,
    MultiRandomApplication,
    MultiPoissonApplication,
)


class TestApplication(unittest.TestCase):
    def test_single_constant_application(self):
        app = SingleConstantApplication("App", "alice", "bob", 1)

        self.assertEqual("App", app.name)

        for timeslot in range(10):
            pairs = app.get_pairs(timeslot)
            self.assertEqual(1, len(pairs))
            self.assertEqual(("alice", "bob", 1), pairs[0])

        app2 = SingleConstantApplication("", "alice", "bob", 2)

        self.assertEqual([("alice", "bob", 2)], app2.get_pairs(0))

        with self.assertRaises(ValueError):
            SingleConstantApplication("", "alice", "alice", 2)

        _ = SingleConstantApplication("", "alice", "bob", 0)

        with self.assertRaises(ValueError):
            SingleConstantApplication("", "alice", "bob", -1)

    def test_single_random_application(self):
        with self.assertRaises(ValueError):
            SingleRandomApplication("App", [], 1)

        with self.assertRaises(ValueError):
            SingleRandomApplication("App", ["alice"], 1)

        _ = SingleRandomApplication("App", ["alice", "bob"], 0)

        with self.assertRaises(ValueError):
            SingleRandomApplication("App", ["alice", "bob"], -1)

        app = SingleRandomApplication("App", ["alice", "bob"], 7)

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

        node_names = set([f"{x}" for x in range(10)])

        app2 = SingleRandomApplication("", node_names, 1)

        found_names = set()
        for timeslot in range(100):
            pair = app2.get_pairs(timeslot)[0]
            found_names.add(pair[0])

        self.assertEqual(node_names, found_names)

    def test_multi_constant_application(self):
        with self.assertRaises(ValueError):
            MultiConstantApplication("App", [], 1)

        with self.assertRaises(ValueError):
            MultiConstantApplication("App", [["alice", "alice"]], 1)

        with self.assertRaises(ValueError):
            MultiConstantApplication(
                "App", [["alice", "bob"], ["alice", "charlie"]], -1
            )

        app = MultiConstantApplication(
            "App", [["alice", "bob"], ["alice", "charlie"]], 7
        )

        for t in range(10):
            self.assertEqual(
                [["alice", "bob", 7], ["alice", "charlie", 7]], app.get_pairs(t)
            )

    def test_multi_random_application(self):
        with self.assertRaises(ValueError):
            MultiRandomApplication("App", [], 2, 1)

        with self.assertRaises(ValueError):
            MultiRandomApplication("App", [["alice", "alice"]], 2, 1)

        with self.assertRaises(ValueError):
            MultiRandomApplication(
                "App", [["alice", "bob"], ["alice", "charlie"]], 2, -1
            )

        with self.assertRaises(ValueError):
            MultiRandomApplication(
                "App", [["alice", "bob"], ["alice", "charlie"]], 3, 1
            )

        app = MultiRandomApplication(
            "App", [["alice", "bob"], ["alice", "charlie"], ["bob", "charlie"]], 2, 7
        )

        everybody = {"alice", "bob", "charlie"}

        found = set()
        for t in range(10):
            pairs = app.get_pairs(t)
            self.assertEqual(2, len(pairs))
            for pair in pairs:
                self.assertEqual(7, pair[2])
                found.add(pair[0])
                found.add(pair[1])

        self.assertEqual(everybody, found)

    def test_multi_poisson_application(self):
        seed = 42

        with self.assertRaises(ValueError):
            MultiPoissonApplication("App", [], 2, 1, seed)

        with self.assertRaises(ValueError):
            MultiPoissonApplication("App", [["alice", "alice"]], 2, 1, seed)

        with self.assertRaises(ValueError):
            MultiPoissonApplication(
                "App", [["alice", "bob"], ["alice", "charlie"]], 2, -1, seed
            )

        app = MultiPoissonApplication(
            "App",
            [["alice", "bob"], ["alice", "charlie"], ["bob", "charlie"]],
            2,
            7,
            seed,
        )

        everybody = {"alice", "bob", "charlie"}

        found = set()
        drawn = set()
        for t in range(100):
            pairs = app.get_pairs(t)
            drawn.add(len(pairs))
            for pair in pairs:
                self.assertEqual(7, pair[2])
                found.add(pair[0])
                found.add(pair[1])

        self.assertEqual(everybody, found)
        for i in range(6):
            self.assertTrue(i in drawn)


if __name__ == "__main__":
    unittest.main()
