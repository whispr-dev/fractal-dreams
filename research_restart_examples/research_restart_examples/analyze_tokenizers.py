from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from analysis_common import build_record_rows, final_record_rows, iter_suite_dirs, load_model_registry_snapshot, extract_generation_stats


def load_tokenizer(hf_id: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)


def token_count(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def analyze_suite(suite_dir: Path, out_dir: Path, reference_tokenizer_hf_id: str | None, include_all_model_tokenizers: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = final_record_rows(build_record_rows(suite_dir))
    registry = load_model_registry_snapshot(suite_dir)

    tokenizer_specs: Dict[str, str] = {}
    if include_all_model_tokenizers:
        for key, meta in registry.items():
            hf_id = meta.get("hf_id")
            if hf_id:
                tokenizer_specs[f"modeltok::{key}"] = hf_id
    if reference_tokenizer_hf_id:
        tokenizer_specs[f"reference::{reference_tokenizer_hf_id}"] = reference_tokenizer_hf_id

    loaded = {name: load_tokenizer(hf_id) for name, hf_id in tokenizer_specs.items()}

    long_rows: List[Dict[str, Any]] = []
    wide_rows: List[Dict[str, Any]] = []
    for row in rows:
        text = row.output_text or ""
        run_stats = extract_generation_stats(row.record)
        base: Dict[str, Any] = {
            "suite_dir": row.suite_dir,
            "run_dir": row.run_dir,
            "engine": row.engine,
            "model_key": row.model_key,
            "prompt_index": row.prompt_index,
            "prompt_text": row.prompt_text,
            "replicate_index": row.replicate_index,
            "run_seed": row.run_seed,
            "final_step": row.step,
            "char_count": len(text),
            "word_count": len(text.split()),
            **run_stats,
        }
        wide = dict(base)
        for tok_name, tok in loaded.items():
            cnt = token_count(tok, text)
            long_rows.append({**base, "tokenizer_name": tok_name, "tokenizer_hf_id": tokenizer_specs[tok_name], "token_count": cnt, "tokens_per_char": cnt / len(text) if text else 0.0, "tokens_per_word": cnt / len(text.split()) if text.split() else 0.0})
            safe_name = tok_name.replace("::", "__").replace("/", "_").replace("-", "_").replace(".", "_")
            wide[f"token_count__{safe_name}"] = cnt
            wide[f"tokens_per_char__{safe_name}"] = cnt / len(text) if text else 0.0
            wide[f"tokens_per_word__{safe_name}"] = cnt / len(text.split()) if text.split() else 0.0
        wide_rows.append(wide)

    long_df = pd.DataFrame(long_rows)
    wide_df = pd.DataFrame(wide_rows)
    long_df.to_csv(out_dir / "tokenizer_counts_long.csv", index=False)
    wide_df.to_csv(out_dir / "tokenizer_counts_wide.csv", index=False)

    summary = long_df.groupby(["engine", "model_key", "tokenizer_name"], dropna=False).agg(
        runs=("run_dir", "count"),
        mean_token_count=("token_count", "mean"),
        sd_token_count=("token_count", "std"),
        mean_tokens_per_char=("tokens_per_char", "mean"),
        mean_tokens_per_word=("tokens_per_word", "mean"),
    ).reset_index()
    summary.to_csv(out_dir / "tokenizer_summary.csv", index=False)

    native = wide_df[["engine", "model_key", "run_dir", "native_output_tokens", "char_count", "word_count"]].copy()
    native["native_tokens_per_char"] = native["native_output_tokens"] / native["char_count"].replace(0, pd.NA)
    native["native_tokens_per_word"] = native["native_output_tokens"] / native["word_count"].replace(0, pd.NA)
    native.to_csv(out_dir / "native_tokenizer_summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenizer-accurate comparison analysis for recursion suite outputs.")
    parser.add_argument("--input", required=True, help="Suite directory or root containing suite directories")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-tokenizer", default="openai-community/gpt2")
    parser.add_argument("--skip-model-tokenizers", action="store_true")
    args = parser.parse_args()

    root = Path(args.input)
    out_root = Path(args.output_dir)
    for suite_dir in iter_suite_dirs(root):
        suite_out = out_root / suite_dir.name
        analyze_suite(suite_dir, suite_out, None if not args.reference_tokenizer else args.reference_tokenizer, include_all_model_tokenizers=not args.skip_model_tokenizers)


if __name__ == "__main__":
    main()
