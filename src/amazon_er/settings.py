"""Configuration loading and validation for modular pipeline stages."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

import yaml

REQUIRED_SECTIONS = (
    "paths",
    "runtime",
    "validation",
    "normalization",
    "candidates",
    "gbdt",
    "reranker",
    "decoder",
)


def _merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _merge(dict(result[key]), value)
        else:
            result[key] = copy.deepcopy(value)
    return result


@dataclass(frozen=True)
class Settings:
    source: Path
    values: dict[str, Any]

    @property
    def project_root(self) -> Path:
        return self.source.parent.parent

    def get(self, dotted: str, default: Any = None) -> Any:
        value: Any = self.values
        for part in dotted.split("."):
            if not isinstance(value, Mapping) or part not in value:
                return default
            value = value[part]
        return value

    def path(self, name: str) -> Path:
        value = self.get(f"paths.{name}")
        if value is None:
            raise KeyError(f"Missing paths.{name}")
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (self.project_root / path).resolve()

    def canonical_json(self) -> str:
        return json.dumps(self.values, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def digest(self) -> str:
        return sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def validate(self) -> None:
        missing = [section for section in REQUIRED_SECTIONS if section not in self.values]
        if missing:
            raise ValueError(f"Missing configuration sections: {', '.join(missing)}")
        folds = int(self.get("validation.folds", 0))
        evaluation_fold = int(self.get("validation.evaluation_fold", -1))
        if folds < 2 or not 0 <= evaluation_fold < folds:
            raise ValueError("validation requires folds >= 2 and evaluation_fold inside that range")
        top_k = int(self.get("candidates.top_k", 0))
        final_cap = int(self.get("candidates.final_cap", 0))
        if top_k <= 0 or final_cap < top_k:
            raise ValueError("candidate top_k must be positive and final_cap must be >= top_k")
        strategy = self.get("decoder.strategy")
        allowed = {"threshold", "threshold_exclusive", "expected_f", "expected_f_exclusive", "pi_exclusive"}
        if strategy not in allowed:
            raise ValueError(f"decoder.strategy must be one of {sorted(allowed)}, got {strategy!r}")


def load_settings(path: str | Path, overrides: Mapping[str, Any] | None = None) -> Settings:
    source = Path(path).expanduser().resolve()
    values = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if overrides:
        values = _merge(values, overrides)
    result = Settings(source=source, values=values)
    result.validate()
    return result

