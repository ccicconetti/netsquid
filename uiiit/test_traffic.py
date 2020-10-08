__author__  = "Claudio Cicconetti"
__version__ = "0.1.0"
__license__ = "MIT"

import unittest
from traffic import SinglePairConstantApplication

class TestApplication(unittest.TestCase):
    def test_single_pair_constant_application(self):
        app = SinglePairConstantApplication("App", "alice", "bob")

        self.assertEqual("App", app.name)

        for timeslot in range(10):
            self.assertEqual(("alice", "bob", 1), app.get_pairs(timeslot))

if __name__ == '__main__':
    unittest.main()