import unittest

from amazon_er.dense_retrieval import record_text


class DenseRetrievalTests(unittest.TestCase):
    def test_record_text_preserves_unicode_and_prefix(self):
        text = record_text(
            {
                "business_name": "  हाईटेक  Services ",
                "business_address": " 12  Main Road ",
                "country": " India ",
            },
            "query",
        )
        self.assertEqual(
            text,
            "query: business name: हाईटेक Services ; address: 12 Main Road ; country: India",
        )


if __name__ == "__main__":
    unittest.main()
