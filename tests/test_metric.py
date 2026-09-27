import unittest

from amazon_er.metric import entity_f05, evaluate


class MetricTests(unittest.TestCase):
    def test_official_example(self):
        self.assertAlmostEqual(
            entity_f05({"S2-00047", "S3-00812"}, {"S2-00047", "S2-00193", "S3-00812"}), 5 / 7
        )

    def test_empty(self):
        self.assertEqual(entity_f05(set(), set()), 1)
        self.assertEqual(entity_f05(set(), {"x"}), 0)
        self.assertEqual(entity_f05({"x"}, set()), 0)

    def test_macro_and_partial(self):
        self.assertAlmostEqual(entity_f05({"a", "b"}, {"a"}), 5 / 6)
        self.assertEqual(evaluate({"a": set(), "b": {"x"}}, {})["macro_f05"], 0.5)
        self.assertEqual(entity_f05({"a"}, {"b"}), 0)

    def test_unknown(self):
        with self.assertRaises(ValueError):
            evaluate({}, {"x": set()})
