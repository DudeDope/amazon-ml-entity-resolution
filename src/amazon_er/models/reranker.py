"""Optional Qwen yes/no reranking and ambiguity-budget selection."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import polars as pl


def ambiguous_queries(scores: pl.DataFrame, fraction: float, max_queries: int) -> pl.DataFrame:
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be between zero and one")
    summary = (
        scores.sort(["q", "score"], descending=[False, True])
        .group_by("q", maintain_order=True)
        .agg(
            pl.col("score").first().alias("top1"),
            pl.col("score").slice(1, 1).first().fill_null(0.0).alias("top2"),
        )
        .with_columns(
            (
                (1.0 - (pl.col("top1") - 0.5).abs() * 2.0).clip(0.0, 1.0)
                + (1.0 - (pl.col("top1") - pl.col("top2"))).clip(0.0, 1.0)
            ).alias("ambiguity")
        )
    )
    budget = min(max_queries, int(summary.height * fraction))
    return summary.sort("ambiguity", descending=True).head(budget).select("q", "ambiguity")


@dataclass
class QwenYesNoReranker:
    model_name: str
    revision: str | None = None
    device: str = "cuda"
    batch_size: int = 8
    max_tokens: int = 512
    instruction: str = "Determine whether the two records describe the same real-world business entity."

    def __post_init__(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional GPU path
            raise RuntimeError("Qwen reranking requires `pip install -e .[gpu]`") from exc
        self.torch = torch
        kwargs = {"trust_remote_code": True}
        if self.revision:
            kwargs["revision"] = self.revision
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side="left", **kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            **kwargs,
        ).to(self.device).eval()
        self.no_id = self.tokenizer.convert_tokens_to_ids("no")
        self.yes_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.prefix = (
            '<|im_start|>system\nJudge whether the Document meets the Query. '
            'Answer only "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        )
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_ids = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_ids = self.tokenizer.encode(self.suffix, add_special_tokens=False)

    def _format(self, query: str, document: str) -> str:
        return f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {document}"

    def _score_batch(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        torch = self.torch
        texts = [self._format(query, document) for query, document in pairs]
        inner = self.max_tokens - len(self.prefix_ids) - len(self.suffix_ids)
        encoded = self.tokenizer(
            texts,
            truncation=True,
            max_length=max(32, inner),
            add_special_tokens=False,
            return_attention_mask=False,
        )
        encoded["input_ids"] = [self.prefix_ids + ids + self.suffix_ids for ids in encoded["input_ids"]]
        encoded = self.tokenizer.pad(encoded, padding=True, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            logits = self.model(**encoded, use_cache=False).logits[:, -1, [self.no_id, self.yes_id]].float()
            return torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

    def score(self, pairs: Iterable[tuple[str, str]]) -> np.ndarray:
        values = list(pairs)
        result: list[float] = []
        start = 0
        batch = self.batch_size
        while start < len(values):
            try:
                score = self._score_batch(values[start : start + batch])
                result.extend(score.tolist())
                start += len(score)
            except self.torch.cuda.OutOfMemoryError:
                if batch <= 1:
                    raise
                batch = max(1, batch // 2)
                gc.collect()
                self.torch.cuda.empty_cache()
        return np.asarray(result, dtype=np.float32)

