import os
import torch
from diffusers import FluxPipeline
from huggingface_hub import login

HF_TOKEN = "REMOVED_HF_TOKEN"
login(token=HF_TOKEN)

pipe = FluxPipeline.from_pretrained("kpsss34/FHDR_Uncensored", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()

# Each iteration appends one of these to evolve the prompt — the "fractal dream" arc
DREAM_MODIFIERS = [
    "panting, stimulated",
    "moaning, cumming",
    "spreading, gaping",
    "lying back, flat out",
    "spreadeagled, leggs akimbo",
    "figers deep, hand between",
    "upright, close up",
    "looking up it, deep in there",
    "staring at, cunt view",
    "sleeping, exhausted",
]


def generate(prompt, seed=0):
    image = pipe(
        prompt,
        height=1024,
        width=1024,
        guidance_scale=4.0,
        num_inference_steps=40,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(seed),
    ).images[0]
    return image


def fractal_dream(seed_prompt, iterations=10, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    current_prompt = seed_prompt
    history = []

    print(f"[0] Seed prompt: {seed_prompt}\n")

    for i in range(iterations):
        modifier = DREAM_MODIFIERS[i % len(DREAM_MODIFIERS)]
        current_prompt = f"{current_prompt}, {modifier}"

        print(f"[{i+1}] {current_prompt}")
        image = generate(current_prompt, seed=i)

        path = os.path.join(output_dir, f"output_{i+1:02d}.png")
        image.save(path)
        print(f"     → saved {path}\n")

        history.append((current_prompt, path))

    return history


if __name__ == "__main__":
    seed = "Photorealistic 8K ultra sharp, cinematic bedroom lighting, Canon EOS R5 50mm f/1.8, shallow depth of field, realistic skin pores and sheen, masterpiece. (Young woman lounging back on crisp white sheets:1.4), (blue t-shirt completely tossed aside on floor:1.3), long dark hair fanned out across pillow, sultry brown eyes locked on viewer with cheeky playful grin, one hand pointing directly at camera like 'your turn next', other hand resting casually on inner thigh. (Legs bent at knees and spread wide open thighs parted knees up:1.5), (perfect explicit unobstructed view straight between her legs:1.4), smooth shaved mound, swollen lips parted, inner pink glistening and visibly gaped open post-fuck stretched and inviting, clit peeking out, fresh wetness shining with max realistic detail and sheen. Signature blue grippy socks with rubbery dotted soles hugging calves and feet, grippy soles facing outward, toes curling playfully. Background standard lamp, radio with soft green glow, fluorescent lights giving naughty warm-cyan ambiance. Large bubbly chaotic fun font title at very top exactly reading 'grippy socks- grippy box!'. Hyper-realistic, best quality, no artifacts, no deformation, no clothes below waist."
    fractal_dream(seed, iterations=10)
