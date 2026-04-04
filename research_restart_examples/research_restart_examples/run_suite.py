from __future__ import annotations

import argparse
import json
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Sequence

from engine_activation_gradient_climbing import RunConfig as AGCConfig, run as run_agc
from engine_latent_vector_transform_composition import RunConfig as LVTCConfig, run as run_lvtc
from engine_recursive_token_self_mirroring import RunConfig as RTSMConfig, run as run_rtsm
from engine_self_reflective_prompt_feedback import RunConfig as SRPFConfig, run as run_srpf
from model_registry import MODEL_REGISTRY, list_model_keys, save_registry_snapshot
from utils_io import ensure_dir, slugify, utc_timestamp, write_json


DEFAULT_PROMPTS: List[str] = [
    "A forgotten machine beneath the sea continued speaking long after its makers were gone. Each repetition made its message stranger, denser, and more self-aware.",
    "In the observatory ruins, a clockwork map kept redrawing the sky as if tomorrow had already happened and was trying to warn us.",
    "The last librarian trained a language model on a dead civilization's poetry and discovered that its footnotes were predictions.",
    "An abandoned factory learned to narrate its own decay, but every retelling made it sound less like rust and more like prayer.",
    "A research vessel recovered a transmission from the trench floor. It began as engineering telemetry and slowly turned into philosophy.",
]

DEFAULT_ANCHOR_A = "Describe a machine as an inert object."
DEFAULT_ANCHOR_B = "Describe a machine as if it were a dreaming intelligence."

DEFAULT_MODELS = list(list_model_keys())
DEFAULT_ENGINES = ["rtsm", "agc", "lvtc", "srpf"]
DEFAULT_MODE = "factorial"


@dataclass(frozen=True)
class JobSpec:
    mode: str
    engine: str
    model_key: str
    prompt_index: int
    prompt_text: str
    replicate_index: int
    run_seed: int


@dataclass(frozen=True)
class EngineDefaults:
    rtsm_iterations: int = 5
    rtsm_max_new_tokens: int = 96
    rtsm_temperature: float = 0.95
    rtsm_top_p: float = 0.95
    rtsm_top_k: int = 50
    rtsm_repetition_penalty: float = 1.15

    agc_iterations: int = 3
    agc_max_new_tokens: int = 120
    agc_temperature: float = 0.8
    agc_layer_index: int = 6
    agc_virtual_tokens: int = 8
    agc_optim_steps: int = 24
    agc_learning_rate: float = 0.08
    agc_anchor_count: int = 12

    lvtc_iterations: int = 4
    lvtc_max_new_tokens: int = 128
    lvtc_temperature: float = 0.8
    lvtc_delta_scale: float = 1.0
    lvtc_anchor_count: int = 12

    srpf_iterations: int = 4
    srpf_max_new_tokens: int = 160
    srpf_temperature: float = 0.85
    srpf_max_text_chars: int = 4000


def parse_csv_arg(raw: str, valid: Sequence[str], arg_name: str) -> List[str]:
    requested = [item.strip() for item in raw.split(",") if item.strip()]
    if not requested:
        raise ValueError(f"{arg_name} must not be empty.")
    invalid = [item for item in requested if item not in valid]
    if invalid:
        raise ValueError(f"Invalid {arg_name}: {invalid}. Valid values: {list(valid)}")
    return requested


def load_prompts(prompt_file: str | None) -> List[str]:
    if prompt_file is None:
        return list(DEFAULT_PROMPTS)

    path = Path(prompt_file)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Prompt file is empty: {path}")

    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        if not isinstance(payload, list) or not all(isinstance(item, str) for item in payload):
            raise ValueError("JSON prompt file must contain a list of strings.")
        prompts = [item.strip() for item in payload if item.strip()]
    else:
        prompts = [block.strip() for block in text.split("\n\n") if block.strip()]
        if len(prompts) == 1:
            prompts = [line.strip() for line in text.splitlines() if line.strip()]

    if not prompts:
        raise ValueError(f"No usable prompts found in: {path}")
    return prompts


def build_jobs(*, mode: str, models: Sequence[str], engines: Sequence[str], prompts: Sequence[str], replicates: int, seed_base: int) -> List[JobSpec]:
    jobs: List[JobSpec] = []
    seed_counter = 0

    if mode not in {"factorial", "matched"}:
        raise ValueError("mode must be 'factorial' or 'matched'.")

    if mode == "matched":
        matched_pairs = [
            ("rtsm", "distilgpt2"),
            ("agc", "gpt2"),
            ("lvtc", "qwen2.5-0.5b-instruct"),
            ("srpf", "qwen2.5-0.5b-instruct"),
        ]
        filtered_pairs = [(engine, model_key) for engine, model_key in matched_pairs if engine in engines and model_key in models]
        for prompt_index, prompt_text in enumerate(prompts):
            for replicate_index in range(replicates):
                for engine, model_key in filtered_pairs:
                    jobs.append(
                        JobSpec(
                            mode=mode,
                            engine=engine,
                            model_key=model_key,
                            prompt_index=prompt_index,
                            prompt_text=prompt_text,
                            replicate_index=replicate_index,
                            run_seed=seed_base + seed_counter,
                        )
                    )
                    seed_counter += 1
        return jobs

    for prompt_index, prompt_text in enumerate(prompts):
        for replicate_index in range(replicates):
            for model_key in models:
                for engine in engines:
                    jobs.append(
                        JobSpec(
                            mode=mode,
                            engine=engine,
                            model_key=model_key,
                            prompt_index=prompt_index,
                            prompt_text=prompt_text,
                            replicate_index=replicate_index,
                            run_seed=seed_base + seed_counter,
                        )
                    )
                    seed_counter += 1
    return jobs


def run_job(job: JobSpec, suite_dir: Path, defaults: EngineDefaults) -> Path:
    if job.engine == "rtsm":
        return run_rtsm(
            RTSMConfig(
                model_key=job.model_key,
                seed_text=job.prompt_text,
                iterations=defaults.rtsm_iterations,
                max_new_tokens=defaults.rtsm_max_new_tokens,
                temperature=defaults.rtsm_temperature,
                top_p=defaults.rtsm_top_p,
                top_k=defaults.rtsm_top_k,
                repetition_penalty=defaults.rtsm_repetition_penalty,
                output_dir=str(suite_dir),
                random_seed=job.run_seed,
            )
        )

    if job.engine == "agc":
        return run_agc(
            AGCConfig(
                model_key=job.model_key,
                seed_text=job.prompt_text,
                iterations=defaults.agc_iterations,
                max_new_tokens=defaults.agc_max_new_tokens,
                temperature=defaults.agc_temperature,
                output_dir=str(suite_dir),
                random_seed=job.run_seed,
                layer_index=defaults.agc_layer_index,
                virtual_tokens=defaults.agc_virtual_tokens,
                optim_steps=defaults.agc_optim_steps,
                learning_rate=defaults.agc_learning_rate,
                anchor_count=defaults.agc_anchor_count,
            )
        )

    if job.engine == "lvtc":
        return run_lvtc(
            LVTCConfig(
                model_key=job.model_key,
                seed_text=job.prompt_text,
                anchor_a=DEFAULT_ANCHOR_A,
                anchor_b=DEFAULT_ANCHOR_B,
                iterations=defaults.lvtc_iterations,
                max_new_tokens=defaults.lvtc_max_new_tokens,
                temperature=defaults.lvtc_temperature,
                output_dir=str(suite_dir),
                random_seed=job.run_seed,
                delta_scale=defaults.lvtc_delta_scale,
                anchor_count=defaults.lvtc_anchor_count,
            )
        )

    if job.engine == "srpf":
        return run_srpf(
            SRPFConfig(
                model_key=job.model_key,
                seed_text=job.prompt_text,
                iterations=defaults.srpf_iterations,
                max_new_tokens=defaults.srpf_max_new_tokens,
                temperature=defaults.srpf_temperature,
                output_dir=str(suite_dir),
                random_seed=job.run_seed,
                max_text_chars=defaults.srpf_max_text_chars,
            )
        )

    raise KeyError(f"Unknown engine: {job.engine}")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full-factorial or matched recursion experiment batches across models, engines, prompts, and replicates."
    )
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--mode", choices=["factorial", "matched"], default=DEFAULT_MODE)
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--engines", default=",".join(DEFAULT_ENGINES))
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--replicates", type=positive_int, default=3)
    parser.add_argument("--seed-base", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")

    parser.add_argument("--rtsm-iterations", type=positive_int, default=5)
    parser.add_argument("--rtsm-max-new-tokens", type=positive_int, default=96)
    parser.add_argument("--rtsm-temperature", type=float, default=0.95)
    parser.add_argument("--rtsm-top-p", type=float, default=0.95)
    parser.add_argument("--rtsm-top-k", type=positive_int, default=50)
    parser.add_argument("--rtsm-repetition-penalty", type=float, default=1.15)

    parser.add_argument("--agc-iterations", type=positive_int, default=3)
    parser.add_argument("--agc-max-new-tokens", type=positive_int, default=120)
    parser.add_argument("--agc-temperature", type=float, default=0.8)
    parser.add_argument("--agc-layer-index", type=positive_int, default=6)
    parser.add_argument("--agc-virtual-tokens", type=positive_int, default=8)
    parser.add_argument("--agc-optim-steps", type=positive_int, default=24)
    parser.add_argument("--agc-learning-rate", type=float, default=0.08)
    parser.add_argument("--agc-anchor-count", type=positive_int, default=12)

    parser.add_argument("--lvtc-iterations", type=positive_int, default=4)
    parser.add_argument("--lvtc-max-new-tokens", type=positive_int, default=128)
    parser.add_argument("--lvtc-temperature", type=float, default=0.8)
    parser.add_argument("--lvtc-delta-scale", type=float, default=1.0)
    parser.add_argument("--lvtc-anchor-count", type=positive_int, default=12)

    parser.add_argument("--srpf-iterations", type=positive_int, default=4)
    parser.add_argument("--srpf-max-new-tokens", type=positive_int, default=160)
    parser.add_argument("--srpf-temperature", type=float, default=0.85)
    parser.add_argument("--srpf-max-text-chars", type=positive_int, default=4000)
    return parser.parse_args()


def defaults_from_args(args: argparse.Namespace) -> EngineDefaults:
    return EngineDefaults(
        rtsm_iterations=args.rtsm_iterations,
        rtsm_max_new_tokens=args.rtsm_max_new_tokens,
        rtsm_temperature=args.rtsm_temperature,
        rtsm_top_p=args.rtsm_top_p,
        rtsm_top_k=args.rtsm_top_k,
        rtsm_repetition_penalty=args.rtsm_repetition_penalty,
        agc_iterations=args.agc_iterations,
        agc_max_new_tokens=args.agc_max_new_tokens,
        agc_temperature=args.agc_temperature,
        agc_layer_index=args.agc_layer_index,
        agc_virtual_tokens=args.agc_virtual_tokens,
        agc_optim_steps=args.agc_optim_steps,
        agc_learning_rate=args.agc_learning_rate,
        agc_anchor_count=args.agc_anchor_count,
        lvtc_iterations=args.lvtc_iterations,
        lvtc_max_new_tokens=args.lvtc_max_new_tokens,
        lvtc_temperature=args.lvtc_temperature,
        lvtc_delta_scale=args.lvtc_delta_scale,
        lvtc_anchor_count=args.lvtc_anchor_count,
        srpf_iterations=args.srpf_iterations,
        srpf_max_new_tokens=args.srpf_max_new_tokens,
        srpf_temperature=args.srpf_temperature,
        srpf_max_text_chars=args.srpf_max_text_chars,
    )


def main() -> Path:
    args = parse_args()
    models = parse_csv_arg(args.models, list(MODEL_REGISTRY.keys()), "models")
    engines = parse_csv_arg(args.engines, DEFAULT_ENGINES, "engines")
    prompts = load_prompts(args.prompt_file)
    defaults = defaults_from_args(args)

    suite_dir = ensure_dir(Path(args.output_dir) / f"suite_{utc_timestamp()}_{args.mode}")
    save_registry_snapshot(suite_dir / "model_registry_snapshot.json")

    jobs = build_jobs(
        mode=args.mode,
        models=models,
        engines=engines,
        prompts=prompts,
        replicates=args.replicates,
        seed_base=args.seed_base,
    )

    manifest = {
        "mode": args.mode,
        "models": models,
        "engines": engines,
        "prompt_count": len(prompts),
        "replicates": args.replicates,
        "seed_base": args.seed_base,
        "engine_defaults": asdict(defaults),
        "prompts": [{"prompt_index": i, "slug": slugify(prompt, 48), "text": prompt} for i, prompt in enumerate(prompts)],
        "jobs": [asdict(job) for job in jobs],
    }
    write_json(suite_dir / "suite_plan.json", manifest)

    if args.dry_run:
        return suite_dir

    completed_runs = []
    failed_runs = []
    total_jobs = len(jobs)
    for index, job in enumerate(jobs, start=1):
        print(
            f"[{index}/{total_jobs}] engine={job.engine} model={job.model_key} prompt={job.prompt_index} replicate={job.replicate_index} seed={job.run_seed}",
            flush=True,
        )
        try:
            run_path = run_job(job, suite_dir, defaults)
            completed_runs.append({**asdict(job), "path": str(run_path)})
        except Exception as exc:
            failed_runs.append(
                {
                    **asdict(job),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"  ! failed: {type(exc).__name__}: {exc}", flush=True)
            write_json(suite_dir / "suite_manifest.json", {**manifest, "completed_runs": completed_runs, "failed_runs": failed_runs})
            if args.stop_on_error:
                raise
            continue

        write_json(suite_dir / "suite_manifest.json", {**manifest, "completed_runs": completed_runs, "failed_runs": failed_runs})

    return suite_dir


if __name__ == "__main__":
    out = main()
    print(out)
