from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from analysis_common import (
    build_record_rows,
    distinct_n,
    extract_generation_stats,
    final_record_rows,
    iter_suite_dirs,
    moving_average_ttr,
    mtld,
    readability_metrics,
    repeated_ngram_fraction,
    shannon_entropy,
    simple_words,
)

try:
    import spacy
except Exception:
    spacy = None


def maybe_load_spacy(model_name: str):
    if spacy is None:
        return None
    try:
        return spacy.load(model_name, disable=["ner"])
    except Exception:
        return None


def tfidf_cos(a: str, b: str) -> float:
    if not a.strip() or not b.strip():
        return 0.0
    vec = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
    mat = vec.fit_transform([a, b])
    return float(cosine_similarity(mat[0:1], mat[1:2])[0, 0])


def spacy_counts(nlp, text: str) -> Dict[str, Any]:
    if nlp is None:
        return {}
    doc = nlp(text)
    pos_counts: Dict[str, int] = {}
    for tok in doc:
        if tok.is_space or tok.is_punct:
            continue
        pos_counts[tok.pos_] = pos_counts.get(tok.pos_, 0) + 1
    total = sum(pos_counts.values()) or 1
    return {
        "pos_noun_ratio": pos_counts.get("NOUN", 0) / total,
        "pos_verb_ratio": pos_counts.get("VERB", 0) / total,
        "pos_adj_ratio": pos_counts.get("ADJ", 0) / total,
        "pos_adv_ratio": pos_counts.get("ADV", 0) / total,
        "pos_pron_ratio": pos_counts.get("PRON", 0) / total,
    }


def analyze_suite(suite_dir: Path, out_dir: Path, spacy_model: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    nlp = maybe_load_spacy(spacy_model)
    all_rows = build_record_rows(suite_dir)
    finals = final_record_rows(all_rows)

    record_rows: List[Dict[str, Any]] = []
    for row in finals:
        text = row.output_text or ""
        words = simple_words(text)
        prompt_words = simple_words(row.prompt_text)
        overlap = len(set(words) & set(prompt_words)) / len(set(prompt_words) or {"_"})
        prev_text = ""
        if row.step > 1:
            prev_candidates = [r for r in all_rows if r.suite_dir == row.suite_dir and r.engine == row.engine and r.model_key == row.model_key and r.prompt_index == row.prompt_index and r.replicate_index == row.replicate_index and r.step == row.step - 1]
            prev_text = prev_candidates[0].output_text if prev_candidates else ""

        data: Dict[str, Any] = {
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
            "word_count": len(words),
            "type_token_ratio": len(set(words)) / len(words) if words else 0.0,
            "mattr_50": moving_average_ttr(words, window=50),
            "mtld": mtld(words),
            "hapax_ratio": sum(1 for w in set(words) if words.count(w) == 1) / len(words) if words else 0.0,
            "word_entropy_bits": shannon_entropy(words),
            "char_entropy_bits": shannon_entropy(list(text)),
            "distinct_1": distinct_n(words, 1),
            "distinct_2": distinct_n(words, 2),
            "distinct_3": distinct_n(words, 3),
            "repeated_bigram_fraction": repeated_ngram_fraction(words, 2),
            "repeated_trigram_fraction": repeated_ngram_fraction(words, 3),
            "repeated_4gram_fraction": repeated_ngram_fraction(words, 4),
            "prompt_tfidf_cosine": tfidf_cos(row.prompt_text, text),
            "previous_step_tfidf_cosine": tfidf_cos(prev_text, text),
            "prompt_lexical_overlap": overlap,
        }
        data.update(readability_metrics(text))
        data.update(extract_generation_stats(row.record))
        data.update(spacy_counts(nlp, text))
        record_rows.append(data)

    df = pd.DataFrame(record_rows)
    df.to_csv(out_dir / "linguistics_final_runs.csv", index=False)

    group_cols = ["engine", "model_key", "prompt_index"]
    summary = df.groupby(group_cols, dropna=False).agg(
        runs=("run_dir", "count"),
        mean_words=("word_count", "mean"),
        sd_words=("word_count", "std"),
        mean_mattr_50=("mattr_50", "mean"),
        mean_mtld=("mtld", "mean"),
        mean_word_entropy_bits=("word_entropy_bits", "mean"),
        mean_distinct_2=("distinct_2", "mean"),
        mean_repeated_trigram_fraction=("repeated_trigram_fraction", "mean"),
        mean_prompt_tfidf_cosine=("prompt_tfidf_cosine", "mean"),
        mean_previous_step_tfidf_cosine=("previous_step_tfidf_cosine", "mean"),
        mean_flesch_reading_ease=("flesch_reading_ease", "mean"),
        mean_flesch_kincaid_grade=("flesch_kincaid_grade", "mean"),
        mean_gunning_fog=("gunning_fog", "mean"),
    ).reset_index()
    summary.to_csv(out_dir / "linguistics_group_summary.csv", index=False)

    overall = df.groupby(["engine", "model_key"], dropna=False).agg(
        runs=("run_dir", "count"),
        mean_words=("word_count", "mean"),
        sd_words=("word_count", "std"),
        mean_mattr_50=("mattr_50", "mean"),
        mean_mtld=("mtld", "mean"),
        mean_word_entropy_bits=("word_entropy_bits", "mean"),
        mean_distinct_2=("distinct_2", "mean"),
        mean_repeated_trigram_fraction=("repeated_trigram_fraction", "mean"),
        mean_prompt_tfidf_cosine=("prompt_tfidf_cosine", "mean"),
        mean_previous_step_tfidf_cosine=("previous_step_tfidf_cosine", "mean"),
        mean_flesch_reading_ease=("flesch_reading_ease", "mean"),
        mean_flesch_kincaid_grade=("flesch_kincaid_grade", "mean"),
        mean_gunning_fog=("gunning_fog", "mean"),
    ).reset_index()
    overall.to_csv(out_dir / "linguistics_overall_summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-grade linguistic analysis for recursion suite outputs.")
    parser.add_argument("--input", required=True, help="Suite directory or root containing suite directories")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    args = parser.parse_args()

    root = Path(args.input)
    out_root = Path(args.output_dir)
    for suite_dir in iter_suite_dirs(root):
        suite_out = out_root / suite_dir.name
        analyze_suite(suite_dir, suite_out, args.spacy_model)


if __name__ == "__main__":
    main()
