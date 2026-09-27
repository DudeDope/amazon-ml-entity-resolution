# Model card

This repository defines a pipeline rather than distributing trained weights. A complete run may
contain lexical retrievers, multilingual embedding models, LightGBM ensembles, a Qwen reranker,
calibration models, and an entity-level decoder.

Business names and addresses can encode sensitive or culturally specific information. Evaluate
performance by country, writing system, source and missingness. Transliteration may merge distinct
native strings, so native-script evidence is always retained and transliteration is an additional
view rather than a replacement.

The system is intended for offline entity resolution. It should not make eligibility, employment,
credit, housing, medical, or legal decisions without domain-specific review and safeguards.

