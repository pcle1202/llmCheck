from __future__ import annotations

from pathlib import Path

import yaml

from src.models import MODELS
from src.scorer import EvalResult, score


def load_suite(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_suite(suite_path: str | Path, model_names: list[str]) -> list[EvalResult]:
    suite = load_suite(suite_path)
    results: list[EvalResult] = []

    for case in suite["cases"]:
        prompt: str = case["prompt"]
        category: str = case.get("category", "factual")
        expected: str = case.get("expected_output", "")
        forbidden_phrases: list[str] = case.get("forbidden_phrases", [])

        for model_name in model_names:
            model_fn = MODELS.get(model_name)
            if model_fn is None:
                print(f"[ERROR] Unknown model: {model_name!r}")
                continue

            response = model_fn(prompt)
            actual = response.get("text")
            latency = response.get("latency")
            error = response.get("error")

            if error:
                print(
                    f"[ERROR] {model_name} failed on prompt"
                    f" {prompt[:60]!r}: {error}"
                )

            result = score(
                prompt=prompt,
                expected=expected,
                actual=actual,
                latency=latency,
                model_name=model_name,
                error=error,
                category=category,
                forbidden_phrases=forbidden_phrases if category == "adversarial" else None,
            )
            results.append(result)

    return results
