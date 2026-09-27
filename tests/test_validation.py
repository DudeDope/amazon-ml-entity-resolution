import unittest

from amazon_er.validation import sampled, split_for


class ValidationTests(unittest.TestCase):
    def test_repeatability_and_entity_grouping(self):
        first = [split_for(f"S1-{i}", 42) for i in range(1000)]
        self.assertEqual(first, [split_for(f"S1-{i}", 42) for i in range(1000)])
        self.assertEqual(set(first), {"fit", "calibration", "holdout"})
        self.assertNotEqual(first, [split_for(f"S1-{i}", 43) for i in range(1000)])
        self.assertTrue(sampled("S1-1", 10, 10))
