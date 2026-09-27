import unittest

from amazon_er.normalization import accent_fold, aggressive, basic, numbers


class NormalizationTests(unittest.TestCase):
    def test_unicode(self):
        self.assertEqual(basic(" ÉCOLE & Co. "), "école and co")
        self.assertEqual(accent_fold("École"), "ecole")
        self.assertTrue(accent_fold("भारत"))
        self.assertEqual(aggressive("Acme Pvt. Ltd."), "acme")
        self.assertEqual(numbers("12B, 560001; 12"), ("12", "560001"))
        self.assertEqual(basic(None), "")
