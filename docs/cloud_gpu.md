# Cloud GPU runbook

Use local NVMe or instance storage for Parquet shards, embeddings, and model caches. Keep only
source data, completed stage artifacts, model files, manifests, and final TSVs on durable object
storage. Network-mounted Drive paths are too slow for inner loops.

## A100 / Colab

```bash
git clone https://github.com/your-org/amazon-ml-entity-resolution.git
cd amazon-ml-entity-resolution
bash scripts/bootstrap_a100.sh
bash scripts/run_cpu_pipeline.sh configs/a100.yaml /content/dataset /content/artifacts /content/output
```

Set `HF_TOKEN` in the platform secret store when a configured model requires authentication. The
Qwen models named in the configuration are public, so a missing token warning alone is harmless.
Copy `artifacts/data`, `artifacts/candidates`, `artifacts/features`, `artifacts/models`, and
`output` after each completed stage if the runtime is preemptible.

## AWS

For SageMaker Studio, attach a GPU space backed by an approved `ml.g5`, `ml.p4d`, or newer GPU
instance and open a terminal inside that space. For EC2, use a Deep Learning AMI and an attached
EBS volume large enough for the source data plus intermediate candidates. Run `nvidia-smi`, then
the bootstrap script. Stop the instance after outputs have been copied to S3.

The CPU pipeline name refers to the structured matcher. With `configs/a100.yaml`, candidate
generation also enables GPU dense embeddings. Reranking is deliberately a separate API so an
experiment can restrict it to ambiguous candidates rather than paying to score the full graph.

## Preflight

Always run these before a paid session:

```bash
amazon-er doctor --config configs/a100.yaml > environment.json
amazon-er smoke --config configs/smoke.yaml
```

The smoke command requires no competition data or model downloads and must end with candidate
recall and macro F-beta equal to `1.0`.
