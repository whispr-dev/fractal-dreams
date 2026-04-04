from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def render_table(df: pd.DataFrame, caption: str, label: str, float_format: str = "%.3f") -> str:
    tex = df.to_latex(index=False, escape=True, longtable=False, float_format=lambda x: float_format % x if isinstance(x, (float, int)) else str(x))
    tex = tex.replace("\\begin{tabular}", f"\\begin{{table}}[t]\n\\centering\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\begin{{tabular}}", 1)
    tex = tex.replace("\\end{tabular}", "\\end{tabular}\n\\end{table}", 1)
    return tex


def main() -> None:
    parser = argparse.ArgumentParser(description="Render publication-ready LaTeX tables from analysis CSV outputs.")
    parser.add_argument("--linguistics-summary", required=True)
    parser.add_argument("--tokenizer-summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    ling = pd.read_csv(args.linguistics_summary)
    tok = pd.read_csv(args.tokenizer_summary)

    ling_cols = [c for c in ["engine", "model_key", "runs", "mean_words", "mean_mattr_50", "mean_mtld", "mean_word_entropy_bits", "mean_distinct_2", "mean_repeated_trigram_fraction", "mean_prompt_tfidf_cosine"] if c in ling.columns]
    tok_cols = [c for c in ["engine", "model_key", "tokenizer_name", "runs", "mean_token_count", "mean_tokens_per_char", "mean_tokens_per_word"] if c in tok.columns]

    text = []
    text.append(render_table(ling[ling_cols], "Linguistic summary by engine and model.", "tab:linguistics_summary"))
    text.append("")
    text.append(render_table(tok[tok_cols], "Tokenizer summary by engine, model, and tokenizer.", "tab:tokenizer_summary"))

    Path(args.output).write_text("\n\n".join(text), encoding="utf-8")


if __name__ == "__main__":
    main()
