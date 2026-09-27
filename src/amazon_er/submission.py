"""Streaming deterministic TSV writer; candidates are exactly scored pairs."""

import csv


class SubmissionWriter:
    def __init__(self, directory):
        directory.mkdir(parents=True, exist_ok=True)
        self.files = [
            (directory / name).open("w", encoding="utf-8", newline="")
            for name in ["matching_results.tsv", "candidate_pairs.tsv"]
        ]
        self.writers = [csv.writer(f, delimiter="\t", lineterminator="\n") for f in self.files]
        self.writers[0].writerow(["source1_entity_id", "matched_entity_ids"])
        self.writers[1].writerow(["source1_entity_id", "candidate_entity_ids"])
        self.seen = set()

    def write(self, entity, matches, candidates):
        if entity in self.seen or not entity.startswith("S1-"):
            raise ValueError("Invalid or duplicate S1 ID")
        self.seen.add(entity)
        if len(matches) != len(set(matches)) or len(candidates) != len(set(candidates)):
            raise ValueError("Duplicate target IDs")
        if not set(matches) <= set(candidates):
            raise ValueError("Match outside final candidate set")
        if any(not x.startswith(("S2-", "S3-")) for x in candidates):
            raise ValueError("Invalid target prefix")
        for writer, values in zip(self.writers, [matches, candidates]):
            writer.writerow([entity, ",".join(sorted(values))])

    def close(self):
        for f in self.files:
            f.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
