from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from model_registry import generate_text, load_model, render_prompt
from utils_io import basic_text_metrics, ensure_dir, slugify, summarize_run_records, utc_timestamp, write_json, write_text


@dataclass
class RunConfig:
    model_key: str
    seed_text: str
    iterations: int
    max_new_tokens: int
    temperature: float
    output_dir: str
    random_seed: int


REFLECTION_TEMPLATE = """Analyze the following text and do two things:
1. State its strongest recurring ideas in 2-4 short bullet points.
2. Produce a revised continuation that makes those ideas more structurally coherent and more surprising.

TEXT:
{text}
"""


def run(config: RunConfig) -> Path:
    loaded = load_model(config.model_key)
    run_dir = ensure_dir(Path(config.output_dir) / f"srpf_{utc_timestamp()}_{slugify(config.seed_text, 32)}")

    current_text = config.seed_text.strip()
    records: List[Dict[str, object]] = []
    text_log: List[str] = [f"[0] seed\n{current_text}\n"]

    for step in range(1, config.iterations + 1):
        reflection_user_text = REFLECTION_TEMPLATE.format(text=current_text)
        reflection_prompt = render_prompt(loaded, reflection_user_text)
        reflection_gen = generate_text(
            loaded,
            reflection_prompt,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            seed=config.random_seed + (step * 2),
        )
        reflection = reflection_gen["continuation_text"] or reflection_gen["full_text"]

        rewrite_prompt = render_prompt(
            loaded,
            (
                "Use the reflection below to write the next evolved version of the text. "
                "Keep it as connected prose, not bullet points.\n\n"
                f"ORIGINAL TEXT:\n{current_text}\n\n"
                f"REFLECTION:\n{reflection}"
            ),
        )
        rewrite_gen = generate_text(
            loaded,
            rewrite_prompt,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            seed=config.random_seed + (step * 2) + 1,
        )
        rewritten = rewrite_gen["continuation_text"] or rewrite_gen["full_text"]
        current_text = rewritten.strip()

        record = {
            "step": step,
            "engine": "self_reflective_prompt_feedback",
            "reflection_text": reflection,
            "output_text": current_text,
            "reflection_generation": reflection_gen,
            "rewrite_generation": rewrite_gen,
            "metrics": basic_text_metrics(current_text),
        }
        records.append(record)
        text_log.append(f"[{step}] reflection\n{reflection}\n")
        text_log.append(f"[{step}] output\n{current_text}\n")

    payload = {
        "engine": "self_reflective_prompt_feedback",
        "config": config.__dict__,
        "model": loaded.to_metadata(),
        "summary": summarize_run_records(records),
        "records": records,
    }
    write_json(run_dir / "run.json", payload)
    write_text(run_dir / "run.txt", text_log)
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-Reflective Prompt Feedback experiment")
    parser.add_argument("--model", dest="model_key", default="qwen2.5-0.5b-instruct")
    parser.add_argument("--seed-text", required=True)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()
    out = run(RunConfig(**vars(args)))
    print(out)
