import csv
import tempfile
import unittest
from pathlib import Path

from amazon_er.blocking import Blocker, build_index
from amazon_er.normalization import prepare


class BlockingTests(unittest.TestCase):
    def test_composite_recovers_target_after_broad_block_cap(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "train").mkdir()
            for source in [2, 3]:
                with (root / "train" / f"train_source{source}.tsv").open(
                    "w", encoding="utf-8", newline=""
                ) as f:
                    writer = csv.writer(f, delimiter="\t")
                    writer.writerow(["entity_id", "business_name", "business_address", "country"])
                    if source == 2:
                        writer.writerows(
                            [
                                ["S2-1", "Acme Industrial", "20 Other Lane", "US"],
                                ["S2-2", "Beda Outlet", "12 East Avenue", "US"],
                                ["S2-3", "Acme Industry", "12 South Lane", "US"],
                            ]
                        )
                    else:
                        writer.writerow(["S3-1", "", "", "France"])
            cfg = {
                "data_dir": str(root),
                "work_dir": str(root),
                "index_batch_size": 10,
                "block_limit": 1,
                "prefilter_k": 10,
            }
            q = prepare(
                {
                    "entity_id": "S1-1",
                    "business_name": "Acme Industries",
                    "business_address": "12 North Road",
                    "country": "US",
                }
            )
            path = build_index(cfg, "train")
            b = Blocker(path, cfg)
            self.assertNotIn("S2-3", {r["entity_id"] for r in b.retrieve(q)[0]})
            b.close()
            cfg["composite_blocks"] = True
            build_index(cfg, "train")
            b = Blocker(path, cfg)
            self.assertIn("S2-3", {r["entity_id"] for r in b.retrieve(q)[0]})
            b.close()

    def test_empty_keys_countries_multimatch_and_restart(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "train").mkdir()
            (root / "artifacts").mkdir()
            for source in [2, 3]:
                with (root / "train" / f"train_source{source}.tsv").open(
                    "w", encoding="utf-8", newline=""
                ) as f:
                    writer = csv.writer(f, delimiter="\t")
                    writer.writerow(["entity_id", "business_name", "business_address", "country"])
                    writer.writerow([f"S{source}-1", "École du Nord", "12 Rue Victor", "France"])
                    writer.writerow([f"S{source}-2", "", "", "US"])
            cfg = {
                "data_dir": str(root),
                "work_dir": str(root),
                "index_batch_size": 10,
                "block_limit": 20,
                "prefilter_k": 10,
            }
            path = build_index(cfg, "train")
            build_index(cfg, "train")
            blocker = Blocker(path, cfg)
            self.assertEqual(blocker.con.execute("SELECT count(*) FROM records").fetchone()[0], 4)
            q = prepare(
                {
                    "entity_id": "S1-1",
                    "business_name": "Ecole du Nord",
                    "business_address": "12 Rue Victor",
                    "country": "France",
                }
            )
            one, _ = blocker.retrieve(q)
            two, _ = blocker.retrieve(q)
            self.assertEqual([r["entity_id"] for r in one], [r["entity_id"] for r in two])
            self.assertEqual({r["entity_id"] for r in one}, {"S2-1", "S3-1"})
            blank = prepare(
                {"entity_id": "S1-2", "business_name": "", "business_address": "", "country": "Mars"}
            )
            self.assertEqual(blocker.retrieve(blank)[0], [])
            blocker.close()
            cfg["composite_blocks"] = True
            build_index(cfg, "train")
            blocker = Blocker(path, cfg)
            result, _ = blocker.retrieve(q)
            self.assertEqual({r["entity_id"] for r in result}, {"S2-1", "S3-1"})
            self.assertTrue(any(r["block_mask"] >= 32 for r in result))
            blocker.close()
