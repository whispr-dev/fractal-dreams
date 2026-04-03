#!/usr/bin/env python3
"""
fsmts-rec.py — Fractal Self-Modifying Text System (Recursive)

A recursive text reflection engine that uses GPT-2 to progressively
expand seed concepts through multiple interpretive lenses (mythological,
philosophical, mathematical, poetic, synthetic) with fractal branching.

Each iteration reflects on the previous output through a rotating lens,
building layered meaning. The operator can inject steering prompts
between iterations (interactive mode) or let it run autonomously.

Outputs a structured log of the dream's evolution.
"""

import os
import sys
import json
import random
import textwrap
from datetime import datetime

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


# ─────────────────────────────────────────────
# GPT-2 setup (auto-downloads on first run)
# ─────────────────────────────────────────────
MODEL_NAME = "gpt2"

print(f"[init] Loading {MODEL_NAME}...")
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model.eval()
print(f"[init] {MODEL_NAME} loaded. Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

# Move to GPU if available
if torch.cuda.is_available():
    model = model.cuda()


# ─────────────────────────────────────────────
# Reflection lenses — each shapes the recursion
# ─────────────────────────────────────────────
LENSES = {
    "mythic": (
        "Retell the following as if it were an ancient myth, a story told around fires "
        "for ten thousand years. Find the archetypal truth hidden inside:\n"
        "\"{text}\"\n\n"
        "The myth begins:"
    ),
    "philosophic": (
        "Examine the following as a philosopher would. What assumptions does it conceal? "
        "What paradox lives at its center? What does it reveal about consciousness?\n"
        "\"{text}\"\n\n"
        "Upon reflection:"
    ),
    "mathematic": (
        "Translate the following into the language of mathematics and pattern. "
        "What structure, symmetry, or transformation does it encode? "
        "What equation or theorem would describe its shape?\n"
        "\"{text}\"\n\n"
        "The pattern reveals:"
    ),
    "poetic": (
        "Distill the following into its purest emotional and sonic essence. "
        "Find the rhythm, the image, the single line that carries all the meaning:\n"
        "\"{text}\"\n\n"
        "The verse:"
    ),
    "synthetic": (
        "The following idea has been examined from many angles. Now synthesize it. "
        "What new concept emerges when all perspectives merge? What discovery hides "
        "in the intersection of myth, logic, mathematics, and feeling?\n"
        "\"{text}\"\n\n"
        "The synthesis:"
    ),
    "dream": (
        "You are dreaming. The following is the residue of waking thought dissolving "
        "into something stranger and more true. Let it transform. Let connections form "
        "that logic would forbid:\n"
        "\"{text}\"\n\n"
        "The dream shifts:"
    ),
}

# Default rotation order for autonomous mode
LENS_SEQUENCE = ["mythic", "philosophic", "dream", "mathematic", "poetic", "synthetic"]


# ─────────────────────────────────────────────
# Core inference
# ─────────────────────────────────────────────
def gpt_reflect(prompt, max_tokens=120, temperature=0.92):
    """
    Run GPT-2 inference on the given prompt.
    Returns the full generated text (prompt + continuation).
    """
    device = next(model.parameters()).device
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=900,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=temperature,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def extract_transform(before, after):
    """
    Extract only the new content the model generated (strip the prompt echo).
    GPT-2 returns prompt + continuation, so we peel off the prompt prefix.
    Falls back to full output if the model didn't echo the prompt verbatim.
    """
    if after.startswith(before):
        return after[len(before):].strip()
    # Sometimes GPT-2 slightly mutates the prompt echo — try finding
    # the last occurrence of a known anchor phrase
    for anchor in ["The myth begins:", "Upon reflection:", "The pattern reveals:",
                    "The verse:", "The synthesis:", "The dream shifts:", "get closer:"]:
        idx = after.rfind(anchor)
        if idx != -1:
            return after[idx + len(anchor):].strip()
    return after.strip()


# ─────────────────────────────────────────────
# Fractal branching
# ─────────────────────────────────────────────
def fractal_branch(text, branch_lenses, depth=0, max_depth=1, max_tokens=80):
    """
    Recursively branch the current text through multiple lenses,
    creating a tree of interpretations (fractal self-similarity).

    Returns a list of branch results:
      [{"lens": str, "depth": int, "text": str, "children": [...]}, ...]
    """
    branches = []
    for lens_name in branch_lenses:
        template = LENSES.get(lens_name, LENSES["dream"])
        prompt = template.format(text=text[:500])  # trim to avoid token overflow
        raw = gpt_reflect(prompt, max_tokens=max_tokens)
        generated = extract_transform(prompt, raw)

        node = {
            "lens": lens_name,
            "depth": depth,
            "text": generated,
            "children": [],
        }

        # Recurse if we haven't hit max depth
        if depth < max_depth and generated.strip():
            # Branch into 2 random lenses at deeper levels
            sub_lenses = random.sample(list(LENSES.keys()), min(2, len(LENSES)))
            node["children"] = fractal_branch(
                generated, sub_lenses, depth + 1, max_depth, max_tokens=60
            )

        branches.append(node)
    return branches


# ─────────────────────────────────────────────
# Main dream loop
# ─────────────────────────────────────────────
def fractal_dream(seed, iterations=5, output_dir="outputs", interactive=False,
                  branch_every=3, branch_depth=1):
    """
    Core recursive dream engine.

    Args:
        seed:          Starting text concept
        iterations:    Number of linear reflection passes
        output_dir:    Where to write logs
        interactive:   If True, pause between iterations for operator input
        branch_every:  Spawn fractal branches every N iterations
        branch_depth:  Max recursion depth for branches (0=disabled)

    Returns:
        history: List of iteration records with full provenance
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(output_dir, f"dream_{timestamp}.txt")
    json_path = os.path.join(output_dir, f"dream_{timestamp}.json")

    current = seed
    history = []
    lens_cycle = list(LENS_SEQUENCE)  # copy so we can rotate

    divider = "═" * 72

    print(f"\n{divider}")
    print(f"  FRACTAL DREAM ENGINE — {timestamp}")
    print(f"  iterations={iterations}  interactive={interactive}")
    print(f"  branch_every={branch_every}  branch_depth={branch_depth}")
    print(f"{divider}\n")
    print(f"[0] SEED:\n{textwrap.fill(current, 72)}\n")

    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"FRACTAL DREAM ENGINE — {timestamp}\n")
        log.write(f"Seed: {seed}\n\n")
        log.write(f"{'─'*72}\n")
        log.write(f"[0] SEED:\n{current}\n\n")

        for i in range(1, iterations + 1):
            # Pick the lens for this iteration (rotate through sequence)
            lens_name = lens_cycle[i % len(lens_cycle)]
            template = LENSES[lens_name]

            # ── Interactive injection ──
            operator_note = ""
            if interactive:
                print(f"{'─'*72}")
                print(f"  Iteration {i}/{iterations} — lens: [{lens_name}]")
                print(f"  Current text preview: {current[:120]}...")
                print(f"{'─'*72}")
                print("  [Enter] to continue | Type text to steer | 'lens:NAME' to switch")
                print(f"  Available lenses: {', '.join(LENSES.keys())}")
                print(f"  'branch' to force branch | 'quit' to stop early")
                try:
                    user_input = input("  > ").strip()
                except (EOFError, KeyboardInterrupt):
                    user_input = ""

                if user_input.lower() == "quit":
                    print("[dream] Operator ended dream early.")
                    log.write(f"\n[{i}] OPERATOR QUIT\n")
                    break

                if user_input.lower().startswith("lens:"):
                    requested = user_input.split(":", 1)[1].strip()
                    if requested in LENSES:
                        lens_name = requested
                        template = LENSES[lens_name]
                        print(f"  → Switched to lens: {lens_name}")
                    else:
                        print(f"  → Unknown lens '{requested}', keeping {lens_name}")

                if user_input.lower() == "branch":
                    print(f"  → Forcing fractal branch at iteration {i}...")
                    branch_lenses = random.sample(list(LENSES.keys()), 3)
                    branches = fractal_branch(current, branch_lenses, max_depth=branch_depth)
                    history.append({
                        "iteration": i,
                        "type": "forced_branch",
                        "lens": "multi",
                        "branches": branches,
                    })
                    _log_branches(log, branches, i)
                    # Pick the richest branch to continue from
                    current = _pick_richest_branch(branches, current)
                    continue

                if user_input and not user_input.lower().startswith("lens:"):
                    operator_note = user_input
                    current = f"{current}\n[operator whispers: {operator_note}]"

            # ── Reflect through current lens ──
            # Feed a trimmed version to avoid token overflow, but keep enough context
            reflection_text = current[-800:] if len(current) > 800 else current
            prompt = template.format(text=reflection_text.strip())
            raw_output = gpt_reflect(prompt, max_tokens=120)
            generated = extract_transform(prompt, raw_output)

            # Guard against empty or degenerate output
            if not generated or len(generated) < 10:
                print(f"[{i}] ⚠ Thin output from [{lens_name}], retrying with dream lens...")
                prompt = LENSES["dream"].format(text=reflection_text.strip())
                raw_output = gpt_reflect(prompt, max_tokens=120, temperature=1.0)
                generated = extract_transform(prompt, raw_output)
                lens_name = "dream (fallback)"

            # ── Record ──
            record = {
                "iteration": i,
                "type": "reflection",
                "lens": lens_name,
                "input_preview": current[:200],
                "output": generated,
                "operator_note": operator_note if operator_note else None,
                "branches": None,
            }

            # ── Fractal branching at intervals ──
            if branch_depth > 0 and i % branch_every == 0:
                print(f"[{i}] ⌁ Fractal branch point — spawning sub-reflections...")
                branch_lenses = random.sample(list(LENSES.keys()), 3)
                branches = fractal_branch(generated, branch_lenses, max_depth=branch_depth)
                record["branches"] = branches
                _log_branches(log, branches, i)

                # Weave branch insights back into the main thread
                branch_summaries = [b["text"][:100] for b in branches if b["text"].strip()]
                if branch_summaries:
                    woven = " | ".join(branch_summaries)
                    generated = f"{generated}\n[echoes from the branches: {woven}]"

            current = generated
            history.append(record)

            # ── Display ──
            print(f"\n[{i}] LENS: [{lens_name}]")
            print(textwrap.fill(generated, 72))
            print()

            # ── Log ──
            log.write(f"{'─'*72}\n")
            log.write(f"[{i}] LENS: [{lens_name}]\n")
            if operator_note:
                log.write(f"    OPERATOR: {operator_note}\n")
            log.write(f"{generated}\n\n")

        # ── Final synthesis pass ──
        print(f"\n{'═'*72}")
        print("  FINAL SYNTHESIS")
        print(f"{'═'*72}\n")

        # Gather key fragments from history for a synthesis prompt
        fragments = [h["output"][:150] for h in history if h.get("output")]
        synthesis_seed = " ... ".join(fragments[-4:])  # last 4 iterations
        synth_prompt = LENSES["synthetic"].format(text=synthesis_seed[:800])
        raw_synth = gpt_reflect(synth_prompt, max_tokens=200)
        synthesis = extract_transform(synth_prompt, raw_synth)

        print(textwrap.fill(synthesis, 72))
        print()

        log.write(f"\n{'═'*72}\n")
        log.write(f"FINAL SYNTHESIS:\n{synthesis}\n")

        history.append({
            "iteration": "final",
            "type": "synthesis",
            "lens": "synthetic",
            "output": synthesis,
        })

    # Save structured JSON log
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump({
            "timestamp": timestamp,
            "seed": seed,
            "iterations": iterations,
            "history": _sanitize_history(history),
        }, jf, indent=2, ensure_ascii=False)

    print(f"\n[dream] Text log:  {log_path}")
    print(f"[dream] JSON log:  {json_path}")
    return history


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def _log_branches(log, branches, iteration, indent=0):
    """Write branch tree to the text log."""
    prefix = "  " * indent
    for b in branches:
        log.write(f"{prefix}  ⌁ [{b['lens']}] d={b['depth']}: {b['text'][:200]}\n")
        if b.get("children"):
            _log_branches(log, b["children"], iteration, indent + 1)


def _pick_richest_branch(branches, fallback):
    """Pick the branch with the longest text output to continue from."""
    if not branches:
        return fallback
    best = max(branches, key=lambda b: len(b.get("text", "")))
    return best["text"] if best["text"].strip() else fallback


def _sanitize_history(history):
    """Make history JSON-serializable (strip any non-serializable objects)."""
    clean = []
    for h in history:
        entry = {}
        for k, v in h.items():
            if isinstance(v, (str, int, float, bool, type(None), list, dict)):
                entry[k] = v
            else:
                entry[k] = str(v)
        clean.append(entry)
    return clean


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Default seeds — evocative concepts that bloom well under recursion
    SEEDS = {
        "consciousness": (
            "A mirror reflecting a mirror. The observer watches itself watching. "
            "Somewhere in the infinite regression, awareness ignites — but which "
            "reflection is the real one? Perhaps the question itself is the answer."
        ),
        "emergence": (
            "A single ant knows nothing. A thousand ants build cathedrals. "
            "Where does the architect live? Not in any one ant, not in the colony, "
            "but in the space between signals — the ghost in the gradient."
        ),
        "time": (
            "The river does not flow. It is the riverbed that moves, sliding "
            "beneath still water. Yesterday is downstream. Tomorrow is the stone "
            "that has not yet been carved by the current."
        ),
        "synthesis": (
            "Mathematics is frozen music. Music is liquid mathematics. "
            "Between the equation and the melody, there is a third thing — "
            "neither number nor sound — that both are trying to describe."
        ),
    }

    import argparse
    parser = argparse.ArgumentParser(
        description="Fractal Self-Modifying Text System (Recursive)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python fsmts-rec.py                           # default seed, autonomous
              python fsmts-rec.py -i                        # interactive mode
              python fsmts-rec.py --seed emergence -n 10    # 10 iterations on 'emergence'
              python fsmts-rec.py --custom "your text here" # custom seed
              python fsmts-rec.py --seed time -i -n 8 --branch-depth 2
        """),
    )
    parser.add_argument("--seed", choices=list(SEEDS.keys()), default="consciousness",
                        help="Named seed concept (default: consciousness)")
    parser.add_argument("--custom", type=str, default=None,
                        help="Custom seed text (overrides --seed)")
    parser.add_argument("-n", "--iterations", type=int, default=5,
                        help="Number of reflection iterations (default: 5)")
    parser.add_argument("-i", "--interactive", action="store_true",
                        help="Interactive mode — steer the dream between iterations")
    parser.add_argument("--branch-every", type=int, default=3,
                        help="Spawn fractal branches every N iterations (default: 3)")
    parser.add_argument("--branch-depth", type=int, default=1,
                        help="Max recursion depth for branches (default: 1, 0=disabled)")
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Output directory for logs (default: outputs)")

    args = parser.parse_args()

    seed_text = args.custom if args.custom else SEEDS[args.seed]

    fractal_dream(
        seed=seed_text,
        iterations=args.iterations,
        output_dir=args.output_dir,
        interactive=args.interactive,
        branch_every=args.branch_every,
        branch_depth=args.branch_depth,
    )