from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from model_registry import generate_text, load_model, render_prompt
from utils_io import basic_text_metrics, ensure_dir, slugify, summarize_run_records, utc_timestamp, write_json, write_text


@dataclass
class RunConfig:
    model_key: str
    seed_text: str
    iterations: int
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    output_dir: str
    random_seed: int


def extract_delta(before: str, generated_payload: Dict[str, Any]) -> str:
    continuation = generated_payload.get("continuation_text", "").strip()
    full_text = generated_payload.get("full_text", "").strip()
    if continuation:
        return continuation
    if full_text.startswith(before):
        return full_text[len(before):].strip()
    return full_text


def run(config: RunConfig) -> Path:
    loaded = load_model(config.model_key)
    run_dir = ensure_dir(Path(config.output_dir) / f"rtsm_{utc_timestamp()}_{slugify(config.seed_text, 32)}")

    current_text = config.seed_text.strip()
    records: List[Dict[str, Any]] = []
    text_log: List[str] = [f"[0] seed\n{current_text}\n"]

    for step in range(1, config.iterations + 1):
        prompt = render_prompt(
            loaded,
            user_text=(
                "Continue the following text. Do not explain your process."
                f"\n\nTEXT:\n{current_text}"
            ),
        )
        gen = generate_text(
            loaded,
            prompt,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=config.repetition_penalty,
            seed=config.random_seed + step,
        )
        delta = extract_delta(current_text, gen)
        current_text = f"{current_text} {delta}".strip()

        record = {
            "step": step,
            "engine": "recursive_token_self_mirroring",
            "prompt": prompt,
            "delta_text": delta,
            "output_text": current_text,
            "generation": gen,
            "metrics": basic_text_metrics(current_text),
        }
        records.append(record)
        text_log.append(f"[{step}] delta\n{delta}\n")
        text_log.append(f"[{step}] output\n{current_text}\n")

    payload = {
        "engine": "recursive_token_self_mirroring",
        "config": config.__dict__,
        "model": loaded.to_metadata(),
        "summary": summarize_run_records(records),
        "records": records,
    }

    write_json(run_dir / "run.json", payload)
    write_text(run_dir / "run.txt", text_log)
    return run_dir


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description="Recursive Token Self-Mirroring experiment")
    parser.add_argument("--model", dest="model_key", default="gpt2")
    parser.add_argument("--seed-text", required=True)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()
    return RunConfig(**vars(args))


if __name__ == "__main__":
    out = run(parse_args())
    print(out)
