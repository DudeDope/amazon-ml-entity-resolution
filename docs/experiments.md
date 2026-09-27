# Experiment protocol

Every experiment records the Git commit, configuration hash, input fingerprints, fold assignment,
candidate recall, macro F0.5, pair precision/recall, singleton accuracy, runtime and peak memory.

## Required comparisons

1. Candidate recall versus candidate budget, overall and by country/script/source.
2. GBDT OOF performance before and after mined negatives.
3. Reranker gain on the same OOF candidate population.
4. Graph-feature ablation with identical folds.
5. Global threshold versus existence, expected-F and cardinality decoders.
6. Mean and standard error across folds or repeated seeds.

Accept a change only when its gain is larger than measurement noise, does not hide a major slice
regression, and fits the available inference budget. Public leaderboard feedback is recorded but
does not select high-dimensional configurations.

