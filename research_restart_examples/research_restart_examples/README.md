# Research Restart Example Bundle

This bundle is a clean, text-only starting point for rebuilding the hallucination-engines / latent-recursion work on a defensible footing.

## Included files

- `model_registry.py` — one loader for the recommended Hugging Face model trio.
- `engine_recursive_token_self_mirroring.py` — baseline recursion engine.
- `engine_activation_gradient_climbing.py` — experimental activation-guided recursion baseline.
- `engine_latent_vector_transform_composition.py` — experimental embedding-delta recursion baseline.
- `engine_self_reflective_prompt_feedback.py` — reflective rewrite recursion engine.
- `run_suite.py` — example suite runner showing how to wire the recommended models into the tests.
- `utils_io.py` — logging and basic defensible text metrics.

## Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install torch transformers accelerate sentencepiece
```

## Download models once

```bash
hf download distilbert/distilgpt2
hf download openai-community/gpt2
hf download Qwen/Qwen2.5-0.5B-Instruct
```

## Run one engine manually

```bash
python engine_recursive_token_self_mirroring.py --model gpt2 --seed-text "A machine dreams in recursive echoes beneath the sea." --iterations 5
```

## Run the bundled example suite

```bash
python run_suite.py --output-dir outputs
```

## Important caveat

`engine_activation_gradient_climbing.py` and `engine_latent_vector_transform_composition.py` are honest restart baselines, not validated claims. They are designed so the project can move from prose to runnable experiments, but they still need real test runs and paper-quality evaluation before they count as results.
