from __future__ import annotations
import json
import re
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List
from utils_io import (
    utc_timestamp,
    ensure_dir,
    slugify,
    write_json,
    write_text,
    basic_text_metrics,
    summarize_run_records,
)

def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def slugify(text: str, max_length: int = 80) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    if len(text) > max_length:
        text = text[:max_length].rstrip("-")
    return text or "run"


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {k: to_jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: str | Path, payload: Any) -> None:
    Path(path).write_text(json.dumps(to_jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def write_text(path: str | Path, lines: Iterable[str]) -> None:
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def basic_text_metrics(text: str) -> Dict[str, Any]:
    tokens = text.split()
    unique = len(set(tokens))
    repeated_bigrams = 0
    seen_bigrams = set()
    for i in range(len(tokens) - 1):
        bigram = (tokens[i], tokens[i + 1])
        if bigram in seen_bigrams:
            repeated_bigrams += 1
        else:
            seen_bigrams.add(bigram)
    return {
        "char_count": len(text),
        "word_count": len(tokens),
        "unique_word_count": unique,
        "unique_word_ratio": (unique / len(tokens)) if tokens else 0.0,
        "repeated_bigram_count": repeated_bigrams,
    }


def summarize_run_records(records: List[Dict[str, Any]], output_key: str = "output_text") -> Dict[str, Any]:
    per_step = []
    for record in records:
        output_text = record.get(output_key, "") or ""
        metrics = basic_text_metrics(output_text)
        per_step.append({
            "step": record.get("step"),
            "lens": record.get("lens"),
            **metrics,
        })
    return {"per_step": per_step, "step_count": len(per_step)}

seed_text = "A forgotten machine beneath the sea continued speaking."
run_name = slugify(seed_text, max_length=40)

run_dir = ensure_dir(Path("outputs") / f"{utc_timestamp()}_{run_name}")

records = [
    {"step": 1, "lens": "mythic", "output_text": "The machine spoke in tides."},
    {"step": 2, "lens": "dream", "output_text": "The machine spoke in tides and recursion."},
]

summary = summarize_run_records(records)

write_json(run_dir / "summary.json", summary)
write_json(run_dir / "records.json", records)

log_lines = [
    f"Run directory: {run_dir}",
    f"Seed text: {seed_text}",
    f"Steps: {len(records)}",
]
write_text(run_dir / "run_log.txt", log_lines)
