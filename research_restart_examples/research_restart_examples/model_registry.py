from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_id: str
    family: str
    approx_params: str
    approx_weights_size: str
    prompt_style: str
    notes: str


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "distilgpt2": ModelSpec(
        key="distilgpt2",
        hf_id="distilbert/distilgpt2",
        family="gpt2",
        approx_params="82M",
        approx_weights_size="353MB safetensors",
        prompt_style="plain",
        notes="Fastest baseline; useful as a collapse-prone small reference model.",
    ),
    "gpt2": ModelSpec(
        key="gpt2",
        hf_id="openai-community/gpt2",
        family="gpt2",
        approx_params="124M",
        approx_weights_size="548MB safetensors",
        prompt_style="plain",
        notes="Direct continuation of the current repo's GPT-2 lineage.",
    ),
    "qwen2.5-0.5b-instruct": ModelSpec(
        key="qwen2.5-0.5b-instruct",
        hf_id="Qwen/Qwen2.5-0.5B-Instruct",
        family="qwen2",
        approx_params="0.49B",
        approx_weights_size="988MB safetensors",
        prompt_style="chat",
        notes="Modern small instruct model; adds a stronger prompt-following contrast.",
    ),
}


@dataclass
class LoadedModel:
    spec: ModelSpec
    tokenizer: PreTrainedTokenizerBase
    model: PreTrainedModel
    device: str
    dtype: str

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "model": asdict(self.spec),
            "device": self.device,
            "dtype": self.dtype,
        }


def get_default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_key: str, device: Optional[str] = None) -> LoadedModel:
    if model_key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model key: {model_key}. Available: {sorted(MODEL_REGISTRY)}")

    spec = MODEL_REGISTRY[model_key]
    resolved_device = device or get_default_device()
    torch_dtype = torch.float16 if resolved_device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(spec.hf_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_id,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    model.eval()
    model.to(resolved_device)

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            model.resize_token_embeddings(len(tokenizer))

    return LoadedModel(
        spec=spec,
        tokenizer=tokenizer,
        model=model,
        device=resolved_device,
        dtype=str(torch_dtype).replace("torch.", ""),
    )


def render_prompt(loaded: LoadedModel, user_text: str, system_text: Optional[str] = None) -> str:
    system_text = system_text or "You are a text-only language model participating in a controlled recursion experiment."

    if loaded.spec.prompt_style == "chat" and hasattr(loaded.tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ]
        return loaded.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return f"{system_text}\n\n{user_text.strip()}"


def generate_text(
    loaded: LoadedModel,
    prompt: str,
    *,
    max_new_tokens: int = 120,
    temperature: float = 0.9,
    top_p: float = 0.95,
    top_k: int = 50,
    repetition_penalty: float = 1.15,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    if seed is not None:
        torch.manual_seed(seed)
        if loaded.device == "cuda":
            torch.cuda.manual_seed_all(seed)

    tokenizer = loaded.tokenizer
    model = loaded.model
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=min(tokenizer.model_max_length, 2048))
    encoded = {k: v.to(loaded.device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    continuation_ids = outputs[0][encoded["input_ids"].shape[1]:]
    continuation_text = tokenizer.decode(continuation_ids, skip_special_tokens=True).strip()

    return {
        "prompt": prompt,
        "full_text": full_text,
        "continuation_text": continuation_text,
        "input_token_count": int(encoded["input_ids"].shape[1]),
        "output_token_count": int(outputs[0].shape[0]),
        "new_token_count": int(continuation_ids.shape[0]),
    }


def save_registry_snapshot(path: str | Path) -> None:
    payload = {key: asdict(spec) for key, spec in MODEL_REGISTRY.items()}
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def list_model_keys() -> Iterable[str]:
    return MODEL_REGISTRY.keys()
