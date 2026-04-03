from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch

from model_registry import generate_text, load_model, render_prompt
from utils_io import basic_text_metrics, ensure_dir, slugify, summarize_run_records, utc_timestamp, write_json, write_text


@dataclass
class RunConfig:
    model_key: str
    seed_text: str
    anchor_a: str
    anchor_b: str
    iterations: int
    max_new_tokens: int
    temperature: float
    output_dir: str
    random_seed: int
    delta_scale: float
    anchor_count: int


def _mean_prompt_embedding(loaded, text: str) -> torch.Tensor:
    tokenizer = loaded.tokenizer
    model = loaded.model
    device = loaded.device
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=min(tokenizer.model_max_length, 256))
    input_ids = encoded["input_ids"].to(device)
    embeds = model.get_input_embeddings()(input_ids).detach()
    return embeds.mean(dim=1).squeeze(0)


def _nearest_anchor_tokens(loaded, vector: torch.Tensor, k: int) -> List[str]:
    vocab = loaded.model.get_input_embeddings().weight.detach()
    vocab_norm = torch.nn.functional.normalize(vocab, dim=-1)
    vec_norm = torch.nn.functional.normalize(vector.unsqueeze(0), dim=-1)
    sims = vec_norm @ vocab_norm.T
    top_ids = torch.topk(sims, k=min(k * 8, vocab.shape[0]), dim=-1).indices.squeeze(0).tolist()

    anchors: List[str] = []
    for token_id in top_ids:
        token = loaded.tokenizer.decode([token_id], skip_special_tokens=True).strip()
        if token and token not in anchors:
            anchors.append(token)
            if len(anchors) >= k:
                break
    return anchors


def run(config: RunConfig) -> Path:
    loaded = load_model(config.model_key)
    run_dir = ensure_dir(Path(config.output_dir) / f"lvtc_{utc_timestamp()}_{slugify(config.seed_text, 32)}")

    anchor_a_vec = _mean_prompt_embedding(loaded, config.anchor_a)
    anchor_b_vec = _mean_prompt_embedding(loaded, config.anchor_b)
    delta = (anchor_b_vec - anchor_a_vec) * config.delta_scale

    current_text = config.seed_text.strip()
    current_vec = _mean_prompt_embedding(loaded, current_text)
    records: List[Dict[str, object]] = []
    text_log: List[str] = [f"[0] seed\n{current_text}\n"]

    for step in range(1, config.iterations + 1):
        current_vec = current_vec + delta
        anchors = _nearest_anchor_tokens(loaded, current_vec, config.anchor_count)
        prompt = render_prompt(
            loaded,
            (
                "Write the next recursive transformation of the text below. "
                "Let the semantic direction be guided by these latent anchor tokens.\n\n"
                f"LATENT DIRECTION TOKENS: {', '.join(anchors)}\n\n"
                f"TEXT:\n{current_text}"
            ),
        )
        gen = generate_text(
            loaded,
            prompt,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            seed=config.random_seed + step,
        )
        output_text = gen["continuation_text"] or gen["full_text"]
        current_text = output_text.strip()
        current_vec = _mean_prompt_embedding(loaded, current_text)

        record = {
            "step": step,
            "engine": "latent_vector_transform_composition",
            "latent_anchor_tokens": anchors,
            "output_text": current_text,
            "generation": gen,
            "metrics": basic_text_metrics(current_text),
        }
        records.append(record)
        text_log.append(f"[{step}] latent direction\n{', '.join(anchors)}\n")
        text_log.append(f"[{step}] output\n{current_text}\n")

    payload = {
        "engine": "latent_vector_transform_composition",
        "config": config.__dict__,
        "model": loaded.to_metadata(),
        "summary": summarize_run_records(records),
        "records": records,
    }
    write_json(run_dir / "run.json", payload)
    write_text(run_dir / "run.txt", text_log)
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latent Vector Transform Composition experiment")
    parser.add_argument("--model", dest="model_key", default="qwen2.5-0.5b-instruct")
    parser.add_argument("--seed-text", required=True)
    parser.add_argument("--anchor-a", required=True)
    parser.add_argument("--anchor-b", required=True)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--delta-scale", type=float, default=1.0)
    parser.add_argument("--anchor-count", type=int, default=12)
    args = parser.parse_args()
    out = run(RunConfig(**vars(args)))
    print(out)
