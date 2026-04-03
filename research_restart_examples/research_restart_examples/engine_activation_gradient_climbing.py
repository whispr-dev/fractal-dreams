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
    iterations: int
    max_new_tokens: int
    temperature: float
    output_dir: str
    random_seed: int
    layer_index: int
    virtual_tokens: int
    optim_steps: int
    learning_rate: float
    anchor_count: int


def _token_embedding_table(loaded):
    return loaded.model.get_input_embeddings().weight.detach()


def _nearest_anchor_tokens(loaded, vectors: torch.Tensor, k: int) -> List[str]:
    vocab = _token_embedding_table(loaded)
    vocab_norm = torch.nn.functional.normalize(vocab, dim=-1)
    vec_norm = torch.nn.functional.normalize(vectors, dim=-1)
    sims = vec_norm @ vocab_norm.T
    top_ids = torch.topk(sims, k=min(k * 4, vocab.shape[0]), dim=-1).indices

    anchors: List[str] = []
    for row in top_ids:
        for token_id in row.tolist():
            token = loaded.tokenizer.decode([token_id], skip_special_tokens=True).strip()
            if token and token not in anchors:
                anchors.append(token)
                if len(anchors) >= k:
                    return anchors
    return anchors[:k]


def _optimize_virtual_prefix(loaded, text: str, layer_index: int, virtual_tokens: int, optim_steps: int, learning_rate: float) -> torch.Tensor:
    tokenizer = loaded.tokenizer
    model = loaded.model
    device = loaded.device

    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=min(tokenizer.model_max_length, 256))
    input_ids = encoded["input_ids"].to(device)
    base_embeds = model.get_input_embeddings()(input_ids).detach()

    seed_slice = base_embeds[:, :virtual_tokens, :]
    if seed_slice.shape[1] < virtual_tokens:
        pad = seed_slice.mean(dim=1, keepdim=True).repeat(1, virtual_tokens - seed_slice.shape[1], 1)
        init_prefix = torch.cat([seed_slice, pad], dim=1)
    else:
        init_prefix = seed_slice.clone()

    prefix = torch.nn.Parameter(init_prefix)
    optimizer = torch.optim.Adam([prefix], lr=learning_rate)
    attention_mask = torch.ones((1, virtual_tokens + input_ids.shape[1]), device=device, dtype=torch.long)

    for _ in range(optim_steps):
        optimizer.zero_grad(set_to_none=True)
        inputs_embeds = torch.cat([prefix, base_embeds], dim=1)
        outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, output_hidden_states=True)
        hidden = outputs.hidden_states[layer_index][:, -1, :]
        objective = hidden.norm(dim=-1).mean()
        loss = -objective
        loss.backward()
        optimizer.step()

    return prefix.detach().squeeze(0)


def run(config: RunConfig) -> Path:
    loaded = load_model(config.model_key)
    run_dir = ensure_dir(Path(config.output_dir) / f"agc_{utc_timestamp()}_{slugify(config.seed_text, 32)}")

    current_text = config.seed_text.strip()
    records: List[Dict[str, object]] = []
    text_log: List[str] = [f"[0] seed\n{current_text}\n"]

    hidden_layer_count = getattr(loaded.model.config, "n_layer", None) or getattr(loaded.model.config, "num_hidden_layers", None)
    if hidden_layer_count is None:
        raise RuntimeError("Could not determine hidden layer count for this model.")
    layer_index = min(config.layer_index, hidden_layer_count)

    for step in range(1, config.iterations + 1):
        prefix_vectors = _optimize_virtual_prefix(
            loaded,
            text=current_text,
            layer_index=layer_index,
            virtual_tokens=config.virtual_tokens,
            optim_steps=config.optim_steps,
            learning_rate=config.learning_rate,
        )
        anchors = _nearest_anchor_tokens(loaded, prefix_vectors, config.anchor_count)
        prompt = render_prompt(
            loaded,
            (
                "Write the next recursive transformation of the text below. "
                "Use the latent anchor tokens only as soft guidance, not as a list to copy verbatim.\n\n"
                f"LATENT ANCHORS: {', '.join(anchors)}\n\n"
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

        record = {
            "step": step,
            "engine": "activation_gradient_climbing",
            "layer_index": layer_index,
            "latent_anchor_tokens": anchors,
            "output_text": current_text,
            "generation": gen,
            "metrics": basic_text_metrics(current_text),
        }
        records.append(record)
        text_log.append(f"[{step}] latent anchors\n{', '.join(anchors)}\n")
        text_log.append(f"[{step}] output\n{current_text}\n")

    payload = {
        "engine": "activation_gradient_climbing",
        "config": config.__dict__,
        "model": loaded.to_metadata(),
        "summary": summarize_run_records(records),
        "records": records,
    }
    write_json(run_dir / "run.json", payload)
    write_text(run_dir / "run.txt", text_log)
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Activation Gradient Climbing experiment")
    parser.add_argument("--model", dest="model_key", default="gpt2")
    parser.add_argument("--seed-text", required=True)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--layer-index", type=int, default=6)
    parser.add_argument("--virtual-tokens", type=int, default=8)
    parser.add_argument("--optim-steps", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--anchor-count", type=int, default=12)
    args = parser.parse_args()
    out = run(RunConfig(**vars(args)))
    print(out)
