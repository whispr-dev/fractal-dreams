from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

WORD_RE = re.compile(r"\b[\w'’-]+\b", re.UNICODE)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class RecordRow:
    suite_dir: str
    run_dir: str
    engine: str
    model_key: str
    prompt_index: int
    prompt_text: str
    replicate_index: int
    run_seed: int
    step: int
    output_text: str
    record: Dict[str, Any]
    run_json: Dict[str, Any]


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def iter_suite_dirs(root: Path) -> Iterable[Path]:
    if root.is_file() and root.name == "suite_manifest.json":
        yield root.parent
        return
    if (root / "suite_manifest.json").exists():
        yield root
    for p in sorted(root.rglob("suite_manifest.json")):
        yield p.parent


def load_suite_manifest(suite_dir: Path) -> Dict[str, Any]:
    return read_json(suite_dir / "suite_manifest.json")


def load_model_registry_snapshot(suite_dir: Path) -> Dict[str, Any]:
    path = suite_dir / "model_registry_snapshot.json"
    return read_json(path) if path.exists() else {}


def build_record_rows(suite_dir: Path) -> List[RecordRow]:
    manifest = load_suite_manifest(suite_dir)
    rows: List[RecordRow] = []
    for run_meta in manifest.get("completed_runs", []):
        path_text = str(run_meta["path"]).replace("\\", "/")
        run_dir = Path(*path_text.split("/"))
        if not run_dir.is_absolute():
            run_dir = suite_dir.parent.parent / run_dir
        run_json_path = run_dir / "run.json"
        if not run_json_path.exists():
            alt = suite_dir / Path(run_meta["path"]).name / "run.json"
            if alt.exists():
                run_json_path = alt
            else:
                continue
        run_json = read_json(run_json_path)
        for record in run_json.get("records", []):
            rows.append(
                RecordRow(
                    suite_dir=str(suite_dir),
                    run_dir=str(run_dir),
                    engine=run_meta["engine"],
                    model_key=run_meta["model_key"],
                    prompt_index=int(run_meta["prompt_index"]),
                    prompt_text=run_meta["prompt_text"],
                    replicate_index=int(run_meta["replicate_index"]),
                    run_seed=int(run_meta["run_seed"]),
                    step=int(record.get("step", 0)),
                    output_text=record.get("output_text", ""),
                    record=record,
                    run_json=run_json,
                )
            )
    return rows


def final_record_rows(rows: List[RecordRow]) -> List[RecordRow]:
    best: Dict[tuple, RecordRow] = {}
    for row in rows:
        key = (row.suite_dir, row.engine, row.model_key, row.prompt_index, row.replicate_index)
        if key not in best or row.step > best[key].step:
            best[key] = row
    return list(best.values())


def simple_words(text: str) -> List[str]:
    return [m.group(0).lower() for m in WORD_RE.finditer(text)]


def simple_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    parts = SENTENCE_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def shannon_entropy(items: List[str]) -> float:
    if not items:
        return 0.0
    counts: Dict[str, int] = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    total = len(items)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def repeated_ngram_fraction(words: List[str], n: int) -> float:
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    total = len(ngrams)
    unique = len(set(ngrams))
    return 1.0 - (unique / total if total else 1.0)


def distinct_n(words: List[str], n: int) -> float:
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    return len(set(ngrams)) / len(ngrams) if ngrams else 0.0


def moving_average_ttr(words: List[str], window: int = 50) -> float:
    if not words:
        return 0.0
    if len(words) <= window:
        return len(set(words)) / len(words)
    vals = []
    for i in range(len(words) - window + 1):
        chunk = words[i:i+window]
        vals.append(len(set(chunk)) / window)
    return sum(vals) / len(vals) if vals else 0.0


def mtld(words: List[str], threshold: float = 0.72) -> float:
    if not words:
        return 0.0

    def _one_pass(seq: List[str]) -> float:
        factors = 0.0
        token_count = 0
        types = set()
        for w in seq:
            token_count += 1
            types.add(w)
            ttr = len(types) / token_count
            if ttr <= threshold:
                factors += 1.0
                token_count = 0
                types = set()
        if token_count > 0:
            ttr = len(types) / token_count if token_count else 1.0
            excess = (1.0 - ttr) / (1.0 - threshold) if ttr < 1.0 else 0.0
            factors += excess
        return len(seq) / factors if factors > 0 else float(len(seq))

    return (_one_pass(words) + _one_pass(list(reversed(words)))) / 2.0


def _estimate_syllables_word(word: str) -> int:
    word = re.sub(r"[^a-z]", "", word.lower())
    if not word:
        return 0
    vowels = "aeiouy"
    syllables = 0
    prev_vowel = False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_vowel:
            syllables += 1
        prev_vowel = is_vowel
    if word.endswith("e") and syllables > 1:
        syllables -= 1
    return max(1, syllables)


def readability_metrics(text: str) -> Dict[str, float]:
    words = simple_words(text)
    sents = simple_sentences(text)
    word_count = len(words)
    sent_count = max(1, len(sents))
    syllables = sum(_estimate_syllables_word(w) for w in words)
    complex_words = sum(1 for w in words if _estimate_syllables_word(w) >= 3)
    asl = word_count / sent_count if sent_count else 0.0
    asw = syllables / word_count if word_count else 0.0
    flesch = 206.835 - 1.015 * asl - 84.6 * asw if word_count else 0.0
    fkgl = 0.39 * asl + 11.8 * asw - 15.59 if word_count else 0.0
    fog = 0.4 * (asl + 100 * (complex_words / word_count)) if word_count else 0.0
    return {
        "sentence_count": float(sent_count),
        "avg_sentence_length_words": asl,
        "avg_syllables_per_word": asw,
        "flesch_reading_ease": flesch,
        "flesch_kincaid_grade": fkgl,
        "gunning_fog": fog,
    }


def extract_generation_stats(record: Dict[str, Any]) -> Dict[str, Any]:
    gen = record.get("generation") or record.get("rewrite_generation") or {}
    return {
        "native_input_tokens": gen.get("input_token_count"),
        "native_new_tokens": gen.get("new_token_count"),
        "native_output_tokens": gen.get("output_token_count"),
        "native_context_limit": gen.get("context_limit"),
        "native_prompt_truncated": gen.get("prompt_was_truncated"),
        "native_requested_max_new_tokens": gen.get("requested_max_new_tokens"),
        "native_effective_max_new_tokens": gen.get("effective_max_new_tokens"),
    }
