from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from engine_activation_gradient_climbing import RunConfig as AGCConfig, run as run_agc
from engine_latent_vector_transform_composition import RunConfig as LVTCConfig, run as run_lvtc
from engine_recursive_token_self_mirroring import RunConfig as RTSMConfig, run as run_rtsm
from engine_self_reflective_prompt_feedback import RunConfig as SRPFConfig, run as run_srpf
from model_registry import MODEL_REGISTRY, save_registry_snapshot
from utils_io import ensure_dir, utc_timestamp, write_json


DEFAULT_SEED = (
    "A forgotten machine beneath the sea continued speaking long after its makers were gone. "
    "Each repetition made its message stranger, denser, and more self-aware."
)

DEFAULT_ANCHOR_A = "Describe a machine as an inert object."
DEFAULT_ANCHOR_B = "Describe a machine as if it were a dreaming intelligence."


def main() -> Path:
    parser = argparse.ArgumentParser(description="Run the restart experiment suite across the recommended models.")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--seed-text", default=DEFAULT_SEED)
    args = parser.parse_args()

    suite_dir = ensure_dir(Path(args.output_dir) / f"suite_{utc_timestamp()}")
    save_registry_snapshot(suite_dir / "model_registry_snapshot.json")

    runs: List[dict] = []

    run1 = run_rtsm(RTSMConfig(
        model_key="distilgpt2",
        seed_text=args.seed_text,
        iterations=5,
        max_new_tokens=96,
        temperature=0.95,
        top_p=0.95,
        top_k=50,
        repetition_penalty=1.15,
        output_dir=str(suite_dir),
        random_seed=args.random_seed,
    ))
    runs.append({"engine": "rtsm", "model": "distilgpt2", "path": str(run1)})

    run2 = run_agc(AGCConfig(
        model_key="gpt2",
        seed_text=args.seed_text,
        iterations=3,
        max_new_tokens=120,
        temperature=0.8,
        output_dir=str(suite_dir),
        random_seed=args.random_seed,
        layer_index=6,
        virtual_tokens=8,
        optim_steps=24,
        learning_rate=0.08,
        anchor_count=12,
    ))
    runs.append({"engine": "agc", "model": "gpt2", "path": str(run2)})

    run3 = run_lvtc(LVTCConfig(
        model_key="qwen2.5-0.5b-instruct",
        seed_text=args.seed_text,
        anchor_a=DEFAULT_ANCHOR_A,
        anchor_b=DEFAULT_ANCHOR_B,
        iterations=4,
        max_new_tokens=128,
        temperature=0.8,
        output_dir=str(suite_dir),
        random_seed=args.random_seed,
        delta_scale=1.0,
        anchor_count=12,
    ))
    runs.append({"engine": "lvtc", "model": "qwen2.5-0.5b-instruct", "path": str(run3)})

    run4 = run_srpf(SRPFConfig(
        model_key="qwen2.5-0.5b-instruct",
        seed_text=args.seed_text,
        iterations=4,
        max_new_tokens=160,
        temperature=0.85,
        output_dir=str(suite_dir),
        random_seed=args.random_seed,
    ))
    runs.append({"engine": "srpf", "model": "qwen2.5-0.5b-instruct", "path": str(run4)})

    write_json(suite_dir / "suite_manifest.json", {"models": list(MODEL_REGISTRY), "runs": runs})
    return suite_dir


if __name__ == "__main__":
    out = main()
    print(out)
