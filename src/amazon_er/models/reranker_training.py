"""Export and optional LoRA fine-tuning utilities for the Qwen yes/no reranker."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl


def export_jsonl(pairs: pl.DataFrame, path: str | Path) -> None:
    required = {"query_text", "target_text", "label"}
    if missing := required - set(pairs.columns):
        raise ValueError(f"Missing reranker training columns: {sorted(missing)}")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in pairs.select("query_text", "target_text", "label").iter_rows(named=True):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def train_lora(
    jsonl_path: str | Path,
    model_name: str,
    output_dir: str | Path,
    revision: str | None = None,
    epochs: float = 1.0,
    batch_size: int = 2,
    gradient_accumulation: int = 16,
) -> None:
    """Fine-tune a causal yes/no model with response-token supervision.

    This function imports the optional GPU stack lazily. It intentionally keeps the training
    surface small; large runs should set distributed/DeepSpeed options through Accelerate.
    """
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    except ImportError as exc:  # pragma: no cover - optional GPU path
        raise RuntimeError("LoRA training requires transformers, datasets, peft and torch") from exc
    kwargs = {"trust_remote_code": True}
    if revision:
        kwargs["revision"] = revision
    tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, **kwargs
    )
    model = get_peft_model(
        model,
        LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM"),
    )
    data = load_dataset("json", data_files=str(jsonl_path), split="train")

    def tokenize(batch):
        prompts = [
            "Determine whether these business records match.\nQuery: " + q + "\nDocument: " + t + "\nAnswer:"
            for q, t in zip(batch["query_text"], batch["target_text"])
        ]
        answers = [" yes" if int(value) else " no" for value in batch["label"]]
        full = [prompt + answer for prompt, answer in zip(prompts, answers)]
        encoded = tokenizer(full, truncation=True, max_length=512, padding="max_length")
        labels = []
        for ids, prompt in zip(encoded["input_ids"], prompts):
            prefix_length = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
            labels.append([-100] * min(prefix_length, len(ids)) + ids[prefix_length:])
        encoded["labels"] = labels
        return encoded

    tokenized = data.map(tokenize, batched=True, remove_columns=data.column_names)
    arguments = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=2e-5,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
    )
    Trainer(model=model, args=arguments, train_dataset=tokenized).train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

