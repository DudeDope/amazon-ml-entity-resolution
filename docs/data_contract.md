# Data contract

## Raw sources

Each source TSV contains exactly:

```text
entity_id<TAB>business_name<TAB>business_address<TAB>country
```

`entity_id` prefixes identify the source: `S1-`, `S2-`, or `S3-`. Training ground truth contains:

```text
source1_entity_id<TAB>matched_entity_ids
```

The second column is an empty string or a comma-separated set of S2/S3 IDs.

## Normalized records

The canonical record schema includes original strings plus `name_norm`, `address_norm`,
`name_core`, `digits`, `script`, `name_roman`, and `address_roman`. Normalization must preserve
row order and never replace an original field.

## Candidates

Candidate Parquet files use:

| Column | Type | Meaning |
|---|---|---|
| `q` | string/int | Source-1 identifier or row ID |
| `t` | string/int | Source-2/3 identifier or row ID |
| `retrieval_source` | string | exact, sparse, dense, transliterated, numeric, etc. |
| `rank` | integer | rank within that retriever and query |
| `retrieval_score` | float | retriever-native score |
| `source_mask` | integer | fused provenance bit mask, when available |
| `fused_score` | float | reciprocal-rank or learned fusion score |

Rows are unique by `(q, t)` after fusion.

## Pair scores

Scored-pair files require `q`, `t`, and `score`. Optional columns include `label`, `fold`,
`country`, `script`, `gbdt_score`, `reranker_score`, `existence_probability`, and graph features.

## Submission

`matching_results.tsv` and `candidate_pairs.tsv` contain one row for every test S1 ID, including
empty predictions. Empty fields are serialized as a trailing tab. Matches must be a subset of
candidates and may contain only S2/S3 IDs.

