import unittest

import pandas as pd

from amazon_er.full_training import select_training_rows


class FullTrainingTests(unittest.TestCase):
    def test_keeps_all_positives_hardest_negatives_and_all_evaluation_rows(self):
        frame = pd.DataFrame(
            [
                {"qid": "q1", "cid": "p1", "label": 1, "split": "fit", "combined_tfidf": 0.2},
                {"qid": "q1", "cid": "n1", "label": 0, "split": "fit", "combined_tfidf": 0.1},
                {"qid": "q1", "cid": "n2", "label": 0, "split": "fit", "combined_tfidf": 0.9},
                {"qid": "q2", "cid": "p2", "label": 1, "split": "fit", "combined_tfidf": 0.4},
                {"qid": "q2", "cid": "n3", "label": 0, "split": "fit", "combined_tfidf": 0.5},
                {
                    "qid": "q3",
                    "cid": "eval1",
                    "label": 0,
                    "split": "holdout",
                    "combined_tfidf": 0.0,
                },
                {
                    "qid": "q3",
                    "cid": "eval2",
                    "label": 1,
                    "split": "holdout",
                    "combined_tfidf": 1.0,
                },
            ]
        )
        selected = select_training_rows(frame, 1)
        pairs = set(zip(selected.qid, selected.cid))
        self.assertEqual(
            pairs,
            {("q1", "p1"), ("q1", "n2"), ("q2", "p2"), ("q2", "n3"), ("q3", "eval1"), ("q3", "eval2")},
        )


if __name__ == "__main__":
    unittest.main()
