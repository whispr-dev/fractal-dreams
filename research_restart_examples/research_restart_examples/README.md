# Analysis toolkit for recursion-suite outputs

This toolkit adds two things your original `utils_io.py` metrics do not provide well enough for a publishable write-up:

1. **Polished linguistic analysis**
2. **Tokenizer-accurate model comparisons**

## Why both are needed

For the paper, do **not** rely only on whitespace word counts or one model's tokenizer counts.

Use:
- **linguistic metrics** to characterize style, lexical diversity, repetition, drift, and readability
- **native tokenizer metrics** to describe how each model internally budgeted the text
- **shared reference tokenizer metrics** to compare outputs across models on a common scale

That last point matters because model-native token counts are *not directly comparable* across different tokenizers.

## Files

- `analysis_common.py` — suite/run loading helpers and shared metrics
- `analyze_linguistics.py` — final-output linguistic analysis
- `analyze_tokenizers.py` — native + cross-tokenizer count analysis
- `render_latex_tables.py` — turns summary CSVs into Overleaf-ready LaTeX tables
- `requirements-analysis.txt` — suggested analysis dependencies

## Suggested analysis stack for the paper

### Linguistic analysis
This script computes:
- character count
- word count
- type-token ratio
- MATTR-50
- MTLD
- hapax ratio
- word entropy
- character entropy
- distinct-1/2/3
- repeated 2/3/4-gram fractions
- prompt TF-IDF cosine similarity
- previous-step TF-IDF cosine similarity
- lexical overlap with the seed prompt
- sentence count
- average sentence length
- Flesch reading ease
- Flesch-Kincaid grade
- Gunning fog
- optional POS ratios via spaCy

### Tokenizer analysis
This script reports:
- native input/new/output token counts already logged by the run JSON
- cross-tokenizer token counts for the final output text under:
  - each model tokenizer in the suite registry
  - one shared reference tokenizer
- tokens per character
- tokens per word

## Recommended reporting practice

For the paper, report both:

1. **native model token counts**
   - useful for runtime/context behavior
2. **shared reference tokenizer counts**
   - useful for apples-to-apples cross-model comparison

Do **not** treat native token counts from different tokenizers as directly comparable.

## Install

```bash
pip install -r requirements-analysis.txt
python -m spacy download en_core_web_sm
```

## Run on your outputs root

```bash
python analyze_linguistics.py --input outputs --output-dir analysis/linguistics

python analyze_tokenizers.py --input outputs --output-dir analysis/tokenizers --reference-tokenizer openai-community/gpt2
```

## Generate LaTeX tables

```bash
python render_latex_tables.py --linguistics-summary analysis/linguistics/suite_20260404T090445Z_factorial/linguistics_overall_summary.csv  --tokenizer-summary analysis/tokenizers/suite_20260404T090445Z_factorial/tokenizer_summary.csv --output analysis/tables.tex
```

## Notes

- `distilgpt2` and `gpt2` share the GPT-2 tokenizer family, so their cross-tokenizer counts may match exactly.
- The scripts analyze the **final output at the final recursion step** for each run. That is usually the cleanest unit for between-run statistics.
- If you want per-step analysis later, extend the scripts to iterate all records rather than final records only.
