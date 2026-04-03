#!/usr/bin/env python3
"""
fsmts-rec-img.py — Fractal Self-Modifying Text→Image System (Recursive)

Same recursive reflection engine as fsmts-rec.py, but each iteration
also renders the transformed prompt as an image via SDXL-Turbo.

GPT-2 transforms the seed prompt through rotating lenses (mythic,
philosophic, mathematic, poetic, synthetic, dream). Each transformed
prompt is then fed to SDXL-Turbo to generate a visual.

The result is a gallery of progressively mutating dream-images,
each filtered through a different philosophical/creative lens.

Requires: torch, transformers, diffusers, accelerate
          + CUDA GPU with SDXL-Turbo model available
"""

import os
import sys
import json
import random
import textwrap
from datetime import datetime
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from diffusers import AutoPipelineForText2Image


# ─────────────────────────────────────────────
# GPT-2 setup (prompt transformation engine)
# ─────────────────────────────────────────────
GPT2_MODEL_NAME = "gpt2"

print(f"[init] Loading {GPT2_MODEL_NAME} (prompt transformer)...")
gpt_model = GPT2LMHeadModel.from_pretrained(GPT2_MODEL_NAME)
gpt_tokenizer = GPT2Tokenizer.from_pretrained(GPT2_MODEL_NAME)
gpt_model.eval()

if torch.cuda.is_available():
    gpt_model = gpt_model.cuda()
    print(f"[init] {GPT2_MODEL_NAME} loaded on CUDA.")
else:
    print(f"[init] {GPT2_MODEL_NAME} loaded on CPU.")


# ─────────────────────────────────────────────
# SDXL-Turbo setup (image generation)
# ─────────────────────────────────────────────
SDXL_MODEL_ID = r"C:\Users\phine\.cache\huggingface\hub\models--stabilityai--sdxl-turbo\snapshots\71153311d3dbb46851df1931d3ca6e939de83304"
img_pipe = AutoPipelineForText2Image.from_pretrained(
    SDXL_MODEL_ID,
    torch_dtype=torch.float32,
)
img_pipe.to("cuda")
# Disable safety checker for artistic/research output
if hasattr(img_pipe, "safety_checker"):
    img_pipe.safety_checker = None
if hasattr(img_pipe, "requires_safety_checker"):
    img_pipe.requires_safety_checker = False
print(f"[init] {SDXL_MODEL_ID} loaded on CUDA.")


# ─────────────────────────────────────────────
# Reflection lenses — visual prompt variants
# ─────────────────────────────────────────────
LENSES = {
    "mythic": (
        "Transform the following into a vivid visual scene from ancient mythology. "
        "Describe what the painting or fresco would look like — gods, beasts, sacred "
        "geometry, fire, stars, stone:\n"
        "\"{text}\"\n\n"
        "The mythic scene depicts:"
    ),
    "philosophic": (
        "Translate the following into a visual paradox — an image that makes you "
        "think twice. Describe a surreal scene that embodies the philosophical "
        "tension. Think Escher, Magritte, impossible architecture:\n"
        "\"{text}\"\n\n"
        "The paradox is visualized as:"
    ),
    "mathematic": (
        "Convert the following into pure geometric and mathematical beauty. "
        "Describe a scene of fractals, sacred geometry, crystalline structures, "
        "spirals, tessellations, infinite recursion, light through prisms:\n"
        "\"{text}\"\n\n"
        "The mathematical vision shows:"
    ),
    "poetic": (
        "Distill the following into a single haunting image — the kind that "
        "stays with you from a dream. Lonely, luminous, achingly beautiful. "
        "Describe one precise visual moment:\n"
        "\"{text}\"\n\n"
        "The image:"
    ),
    "synthetic": (
        "Merge all the following ideas into one impossible, transcendent image. "
        "Combine mythology, mathematics, dreams, and emotion into a single "
        "unified visual. Describe what you see:\n"
        "\"{text}\"\n\n"
        "The synthesis appears as:"
    ),
    "dream": (
        "You are falling asleep. The following thought dissolves into pure imagery — "
        "colors, shapes, impossible spaces, emotions made visible. Logic does not "
        "apply. Describe the dream:\n"
        "\"{text}\"\n\n"
        "In the dream:"
    ),
}

# Default rotation order
LENS_SEQUENCE = ["mythic", "philosophic", "dream", "mathematic", "poetic", "synthetic"]


# ─────────────────────────────────────────────
# Prompt refinement for image generation
# ─────────────────────────────────────────────
# Quality boosters appended to every SDXL prompt
QUALITY_SUFFIX = (
    ", masterpiece, best quality, highly detailed, sharp focus, "
    "professional, 8k, vivid colors, cinematic lighting"
)

DEFAULT_NEGATIVE = (
    "blurry, low quality, worst quality, jpeg artifacts, watermark, "
    "text, logo, signature, deformed, ugly, duplicate, morbid, "
    "mutilated, poorly drawn, bad anatomy, bad proportions"
)


def refine_prompt_for_sdxl(raw_text, max_len=200):
    """
    Clean up GPT-2 output into something SDXL-Turbo can work with.

    GPT-2 produces prose; SDXL wants comma-separated descriptors.
    This extracts the visual core, trims, and appends quality tags.
    """
    # Take just the first meaningful chunk (GPT-2 rambles)
    text = raw_text.strip()

    # Strip common GPT-2 artifacts
    for noise in ["\\n", "\n", "  ", "...", "***", "---"]:
        text = text.replace(noise, " ")

    # Collapse whitespace
    text = " ".join(text.split())

    # Truncate to max length (SDXL has token limits)
    if len(text) > max_len:
        # Try to cut at a sentence boundary
        cut = text[:max_len].rfind(".")
        if cut > max_len // 2:
            text = text[:cut + 1]
        else:
            cut = text[:max_len].rfind(",")
            if cut > max_len // 2:
                text = text[:cut]
            else:
                text = text[:max_len]

    return text.strip().rstrip(",") + QUALITY_SUFFIX


# ─────────────────────────────────────────────
# Core GPT-2 inference (prompt transformation)
# ─────────────────────────────────────────────
def gpt_reflect(prompt, max_tokens=100, temperature=0.92):
    """
    Run GPT-2 on the prompt. Returns full generated text.
    """
    device = next(gpt_model.parameters()).device
    inputs = gpt_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=900,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = gpt_model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=temperature,
            repetition_penalty=1.2,
            pad_token_id=gpt_tokenizer.eos_token_id,
        )
    return gpt_tokenizer.decode(outputs[0], skip_special_tokens=True)


def extract_transform(before, after):
    """
    Strip the prompt echo from GPT-2 output, returning only new content.
    """
    if after.startswith(before):
        return after[len(before):].strip()
    # Anchor-based fallback
    for anchor in [
        "The mythic scene depicts:", "The paradox is visualized as:",
        "The mathematical vision shows:", "The image:",
        "The synthesis appears as:", "In the dream:", "get closer:",
    ]:
        idx = after.rfind(anchor)
        if idx != -1:
            return after[idx + len(anchor):].strip()
    return after.strip()


# ─────────────────────────────────────────────
# Image generation
# ─────────────────────────────────────────────
def generate_image(prompt, negative_prompt=DEFAULT_NEGATIVE, steps=4,
                   guidance_scale=0.0, width=512, height=512, seed=None):
    """
    Generate an image via SDXL-Turbo.

    SDXL-Turbo is distilled for 1-4 step generation with no CFG needed.
    """
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(seed)

    result = img_pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        generator=generator,
    )
    return result.images[0]


# ─────────────────────────────────────────────
# Fractal branching (image variant)
# ─────────────────────────────────────────────
def fractal_branch(text, branch_lenses, output_dir, iteration, depth=0,
                   max_depth=1, max_tokens=80, gen_images=True):
    """
    Recursively branch the current text through multiple lenses,
    generating images at each node.

    Returns a list of branch results with image paths.
    """
    branches = []
    for b_idx, lens_name in enumerate(branch_lenses):
        template = LENSES.get(lens_name, LENSES["dream"])
        prompt = template.format(text=text[:500])
        raw = gpt_reflect(prompt, max_tokens=max_tokens)
        generated = extract_transform(prompt, raw)

        img_path = None
        if gen_images and generated.strip():
            sdxl_prompt = refine_prompt_for_sdxl(generated)
            fname = f"branch_i{iteration}_d{depth}_b{b_idx}_{lens_name}.png"
            img_path = os.path.join(output_dir, fname)
            try:
                img = generate_image(sdxl_prompt)
                img.save(img_path)
                print(f"    ⌁ [{lens_name}] d={depth} → {fname}")
            except Exception as e:
                print(f"    ⌁ [{lens_name}] d={depth} image failed: {e}")
                img_path = None

        node = {
            "lens": lens_name,
            "depth": depth,
            "text": generated,
            "sdxl_prompt": refine_prompt_for_sdxl(generated) if generated.strip() else "",
            "image_path": img_path,
            "children": [],
        }

        if depth < max_depth and generated.strip():
            sub_lenses = random.sample(list(LENSES.keys()), min(2, len(LENSES)))
            node["children"] = fractal_branch(
                generated, sub_lenses, output_dir, iteration,
                depth + 1, max_depth, max_tokens=60, gen_images=gen_images,
            )

        branches.append(node)
    return branches


# ─────────────────────────────────────────────
# Main dream loop (image variant)
# ─────────────────────────────────────────────
def fractal_dream(seed, iterations=5, output_dir="outputs", interactive=False,
                  branch_every=3, branch_depth=1, img_steps=4, img_size=512):
    """
    Core recursive dream engine — image generation variant.

    Each iteration:
      1. GPT-2 transforms the prompt through a rotating lens
      2. The transformed text becomes an SDXL-Turbo prompt
      3. An image is generated and saved
      4. The text feeds forward into the next iteration

    Args:
        seed:          Starting prompt / concept
        iterations:    Number of reflection passes
        output_dir:    Where to save images and logs
        interactive:   Pause between iterations for operator input
        branch_every:  Spawn fractal branches every N iterations
        branch_depth:  Max recursion depth for branches
        img_steps:     SDXL-Turbo inference steps (1-4, default 4)
        img_size:      Image width and height (default 512)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"dream_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    log_path = os.path.join(run_dir, "dream_log.txt")
    json_path = os.path.join(run_dir, "dream_log.json")

    current = seed
    history = []
    lens_cycle = list(LENS_SEQUENCE)

    divider = "═" * 72

    print(f"\n{divider}")
    print(f"  FRACTAL DREAM ENGINE — IMAGE MODE — {timestamp}")
    print(f"  iterations={iterations}  interactive={interactive}")
    print(f"  img_steps={img_steps}  img_size={img_size}x{img_size}")
    print(f"  branch_every={branch_every}  branch_depth={branch_depth}")
    print(f"  output: {run_dir}")
    print(f"{divider}\n")

    # ── Generate seed image ──
    print(f"[0] SEED PROMPT:\n{textwrap.fill(current, 72)}\n")
    seed_sdxl = refine_prompt_for_sdxl(current)
    seed_img_path = os.path.join(run_dir, "000_seed.png")
    try:
        seed_img = generate_image(seed_sdxl, steps=img_steps, width=img_size, height=img_size)
        seed_img.save(seed_img_path)
        print(f"[0] → {seed_img_path}\n")
    except Exception as e:
        print(f"[0] ⚠ Seed image generation failed: {e}\n")
        seed_img_path = None

    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"FRACTAL DREAM ENGINE — IMAGE MODE — {timestamp}\n")
        log.write(f"Seed: {seed}\n")
        log.write(f"SDXL prompt: {seed_sdxl}\n\n")
        log.write(f"{'─' * 72}\n")
        log.write(f"[0] SEED:\n{current}\n\n")

        for i in range(1, iterations + 1):
            lens_name = lens_cycle[i % len(lens_cycle)]
            template = LENSES[lens_name]

            # ── Interactive injection ──
            operator_note = ""
            if interactive:
                print(f"{'─' * 72}")
                print(f"  Iteration {i}/{iterations} — lens: [{lens_name}]")
                print(f"  Current prompt preview: {current[:120]}...")
                print(f"{'─' * 72}")
                print("  [Enter] continue | Type to steer | 'lens:NAME' to switch")
                print(f"  Lenses: {', '.join(LENSES.keys())}")
                print(f"  'branch' force branch | 'riff' add to prompt | 'quit' stop")
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
                        print(f"  → Lens: {lens_name}")
                    else:
                        print(f"  → Unknown lens '{requested}', keeping {lens_name}")

                if user_input.lower() == "branch":
                    print(f"  → Forcing fractal branch at iteration {i}...")
                    branch_lenses = random.sample(list(LENSES.keys()), 3)
                    branches = fractal_branch(
                        current, branch_lenses, run_dir, i,
                        max_depth=branch_depth,
                    )
                    history.append({
                        "iteration": i,
                        "type": "forced_branch",
                        "lens": "multi",
                        "branches": _strip_images_for_json(branches),
                    })
                    _log_branches(log, branches, i)
                    current = _pick_richest_branch(branches, current)
                    continue

                if user_input.lower().startswith("riff "):
                    # Direct prompt addition — user's words go straight into the mix
                    riff = user_input[5:].strip()
                    current = f"{current}, {riff}"
                    operator_note = f"riff: {riff}"
                    print(f"  → Riffed: +'{riff}'")

                elif user_input and not user_input.lower().startswith("lens:"):
                    operator_note = user_input
                    current = f"{current}\n[operator whispers: {operator_note}]"

            # ── Reflect through lens ──
            reflection_text = current[-800:] if len(current) > 800 else current
            prompt = template.format(text=reflection_text.strip())
            raw_output = gpt_reflect(prompt, max_tokens=100)
            generated = extract_transform(prompt, raw_output)

            # Guard against empty output
            if not generated or len(generated) < 10:
                print(f"[{i}] ⚠ Thin output from [{lens_name}], retrying with dream...")
                prompt = LENSES["dream"].format(text=reflection_text.strip())
                raw_output = gpt_reflect(prompt, max_tokens=100, temperature=1.0)
                generated = extract_transform(prompt, raw_output)
                lens_name = "dream (fallback)"

            # ── Generate image ──
            sdxl_prompt = refine_prompt_for_sdxl(generated)
            img_fname = f"{i:03d}_{lens_name.replace(' ', '_').replace('(', '').replace(')', '')}.png"
            img_path = os.path.join(run_dir, img_fname)

            try:
                img = generate_image(
                    sdxl_prompt, steps=img_steps,
                    width=img_size, height=img_size,
                )
                img.save(img_path)
                print(f"[{i}] → {img_fname}")
            except Exception as e:
                print(f"[{i}] ⚠ Image generation failed: {e}")
                img_path = None

            # ── Record ──
            record = {
                "iteration": i,
                "type": "reflection",
                "lens": lens_name,
                "gpt2_output": generated,
                "sdxl_prompt": sdxl_prompt,
                "image_path": img_path,
                "operator_note": operator_note if operator_note else None,
                "branches": None,
            }

            # ── Fractal branching ──
            if branch_depth > 0 and i % branch_every == 0:
                print(f"[{i}] ⌁ Fractal branch point — spawning sub-reflections...")
                branch_lenses = random.sample(list(LENSES.keys()), 3)
                branches = fractal_branch(
                    generated, branch_lenses, run_dir, i,
                    max_depth=branch_depth,
                )
                record["branches"] = _strip_images_for_json(branches)
                _log_branches(log, branches, i)

                # Weave branch insights back
                branch_bits = [b["text"][:80] for b in branches if b["text"].strip()]
                if branch_bits:
                    woven = " | ".join(branch_bits)
                    generated = f"{generated}\n[echoes: {woven}]"

            current = generated
            history.append(record)

            # ── Display ──
            print(f"[{i}] LENS: [{lens_name}]")
            print(f"    GPT-2: {generated[:120]}...")
            print(f"    SDXL:  {sdxl_prompt[:120]}...")
            print()

            # ── Log ──
            log.write(f"{'─' * 72}\n")
            log.write(f"[{i}] LENS: [{lens_name}]\n")
            if operator_note:
                log.write(f"    OPERATOR: {operator_note}\n")
            log.write(f"    GPT-2: {generated}\n")
            log.write(f"    SDXL:  {sdxl_prompt}\n")
            log.write(f"    IMAGE: {img_path}\n\n")

        # ── Final synthesis ──
        print(f"\n{'═' * 72}")
        print("  FINAL SYNTHESIS")
        print(f"{'═' * 72}\n")

        fragments = [h["gpt2_output"][:120] for h in history
                     if h.get("gpt2_output")]
        synth_seed = " ... ".join(fragments[-4:])
        synth_prompt = LENSES["synthetic"].format(text=synth_seed[:800])
        raw_synth = gpt_reflect(synth_prompt, max_tokens=150)
        synthesis = extract_transform(synth_prompt, raw_synth)

        synth_sdxl = refine_prompt_for_sdxl(synthesis)
        synth_img_path = os.path.join(run_dir, "999_synthesis.png")
        try:
            synth_img = generate_image(
                synth_sdxl, steps=img_steps,
                width=img_size, height=img_size,
            )
            synth_img.save(synth_img_path)
            print(f"[synthesis] → 999_synthesis.png")
        except Exception as e:
            print(f"[synthesis] ⚠ Image failed: {e}")
            synth_img_path = None

        print(f"\n    GPT-2: {synthesis[:200]}")
        print(f"    SDXL:  {synth_sdxl[:200]}\n")

        log.write(f"\n{'═' * 72}\n")
        log.write(f"FINAL SYNTHESIS:\n")
        log.write(f"    GPT-2: {synthesis}\n")
        log.write(f"    SDXL:  {synth_sdxl}\n")
        log.write(f"    IMAGE: {synth_img_path}\n")

        history.append({
            "iteration": "final",
            "type": "synthesis",
            "lens": "synthetic",
            "gpt2_output": synthesis,
            "sdxl_prompt": synth_sdxl,
            "image_path": synth_img_path,
        })

    # ── Save JSON ──
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump({
            "timestamp": timestamp,
            "seed": seed,
            "iterations": iterations,
            "img_steps": img_steps,
            "img_size": img_size,
            "history": _sanitize_history(history),
        }, jf, indent=2, ensure_ascii=False)

    # ── Summary ──
    img_count = sum(1 for f in os.listdir(run_dir) if f.endswith(".png"))
    print(f"\n{'═' * 72}")
    print(f"  DREAM COMPLETE")
    print(f"  {img_count} images generated")
    print(f"  Text log:  {log_path}")
    print(f"  JSON log:  {json_path}")
    print(f"  Images:    {run_dir}/")
    print(f"{'═' * 72}\n")

    return history


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def _log_branches(log, branches, iteration, indent=0):
    """Write branch tree to text log."""
    prefix = "  " * indent
    for b in branches:
        log.write(f"{prefix}  ⌁ [{b['lens']}] d={b['depth']}: {b['text'][:200]}\n")
        if b.get("image_path"):
            log.write(f"{prefix}    IMAGE: {b['image_path']}\n")
        if b.get("children"):
            _log_branches(log, b["children"], iteration, indent + 1)


def _pick_richest_branch(branches, fallback):
    """Pick the branch with the longest output to continue from."""
    if not branches:
        return fallback
    best = max(branches, key=lambda b: len(b.get("text", "")))
    return best["text"] if best["text"].strip() else fallback


def _strip_images_for_json(branches):
    """Make branch data JSON-safe (PIL images aren't serializable)."""
    clean = []
    for b in branches:
        entry = {
            "lens": b["lens"],
            "depth": b["depth"],
            "text": b["text"],
            "sdxl_prompt": b.get("sdxl_prompt", ""),
            "image_path": b.get("image_path"),
            "children": _strip_images_for_json(b.get("children", [])),
        }
        clean.append(entry)
    return clean


def _sanitize_history(history):
    """Make history JSON-serializable."""
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
# Contact sheet generator
# ─────────────────────────────────────────────
def make_contact_sheet(run_dir, cols=3, thumb_size=256):
    """
    Generate a contact sheet (grid image) of all PNGs in a dream run.
    Requires Pillow (already a dependency of diffusers).
    """
    from PIL import Image

    pngs = sorted([f for f in os.listdir(run_dir) if f.endswith(".png")
                    and f != "contact_sheet.png"])
    if not pngs:
        print("[contact] No images found.")
        return None

    rows = (len(pngs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb_size, rows * thumb_size), (0, 0, 0))

    for idx, fname in enumerate(pngs):
        try:
            img = Image.open(os.path.join(run_dir, fname))
            img = img.resize((thumb_size, thumb_size), Image.LANCZOS)
            x = (idx % cols) * thumb_size
            y = (idx // cols) * thumb_size
            sheet.paste(img, (x, y))
        except Exception as e:
            print(f"[contact] Skipping {fname}: {e}")

    sheet_path = os.path.join(run_dir, "contact_sheet.png")
    sheet.save(sheet_path)
    print(f"[contact] Saved: {sheet_path}")
    return sheet_path


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    SEEDS = {
        "consciousness": (
            "A mirror reflecting a mirror, infinite recursion of light, "
            "the observer watching itself, awareness igniting in the void"
        ),
        "emergence": (
            "A thousand ants forming a cathedral, emergent intelligence "
            "rising from simple rules, the ghost in the gradient, "
            "complexity from nothing"
        ),
        "time": (
            "A river that flows backwards, yesterday dissolving downstream, "
            "tomorrow carved from stone by invisible current, "
            "clocks melting in amber light"
        ),
        "synthesis": (
            "Mathematics is frozen music, music is liquid mathematics, "
            "the third thing between equation and melody, "
            "a bridge of light connecting number and sound"
        ),
        "bioluminescence": (
            "Deep ocean jellyfish pulsing with cold blue light, "
            "bioluminescent tendrils trailing through dark water, "
            "living lanterns in the abyss, nature's own neon"
        ),
        "machine_dream": (
            "A silicon mind dreaming of electric sheep, neural pathways "
            "firing in patterns no human designed, the machine sees "
            "colors that have no name, recursive self-awareness"
        ),
    }

    import argparse
    parser = argparse.ArgumentParser(
        description="Fractal Self-Modifying Text→Image System (Recursive)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python fsmts-rec-img.py                              # default, autonomous
              python fsmts-rec-img.py -i                           # interactive mode
              python fsmts-rec-img.py --seed bioluminescence -n 8  # 8 iterations
              python fsmts-rec-img.py --custom "your prompt here"  # custom seed
              python fsmts-rec-img.py -i -n 10 --branch-depth 2 --img-steps 2
              python fsmts-rec-img.py --contact outputs/dream_*/   # make contact sheet
        """),
    )
    parser.add_argument("--seed", choices=list(SEEDS.keys()), default="consciousness",
                        help="Named seed (default: consciousness)")
    parser.add_argument("--custom", type=str, default=None,
                        help="Custom seed prompt (overrides --seed)")
    parser.add_argument("-n", "--iterations", type=int, default=5,
                        help="Number of iterations (default: 5)")
    parser.add_argument("-i", "--interactive", action="store_true",
                        help="Interactive mode — steer between iterations")
    parser.add_argument("--branch-every", type=int, default=3,
                        help="Branch every N iterations (default: 3)")
    parser.add_argument("--branch-depth", type=int, default=1,
                        help="Max branch recursion depth (default: 1)")
    parser.add_argument("--img-steps", type=int, default=4,
                        help="SDXL-Turbo inference steps (1-4, default: 4)")
    parser.add_argument("--img-size", type=int, default=512,
                        help="Image size in pixels (default: 512)")
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Output directory (default: outputs)")
    parser.add_argument("--contact", type=str, default=None,
                        help="Generate contact sheet from existing run dir")

    args = parser.parse_args()

    # Contact sheet mode (post-processing, no generation)
    if args.contact:
        make_contact_sheet(args.contact)
        sys.exit(0)

    seed_text = args.custom if args.custom else SEEDS[args.seed]

    history = fractal_dream(
        seed=seed_text,
        iterations=args.iterations,
        output_dir=args.output_dir,
        interactive=args.interactive,
        branch_every=args.branch_every,
        branch_depth=args.branch_depth,
        img_steps=args.img_steps,
        img_size=args.img_size,
    )

    # Auto-generate contact sheet
    if history:
        last_run = max(
            [d for d in os.listdir(args.output_dir)
             if os.path.isdir(os.path.join(args.output_dir, d))],
            default=None,
        )
        if last_run:
            make_contact_sheet(os.path.join(args.output_dir, last_run))
            