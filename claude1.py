import os
import torch
from diffusers import AutoPipelineForText2Image

# --- GPT2 setup (auto-downloads, tiny, no wait) ---
gpt_model.eval()

# --- SDXL-Turbo setup ---
pipe = pipe.to("cuda")  # ← this line must be there, NOT commented out

# pipe.enable_model_cpu_offload()

def gpt_reflect(prompt, max_tokens=80):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=900)
    with torch.no_grad():
        outputs = gpt_model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def text_to_text(text):
    # Trim GPT2's verbose output down to a usable image prompt
    return text.strip()[:max_len]

def text_to_text_prompt(text):
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "grippy_sox.txt")
    current = seed
    history = []

    print(f"[0] SEED:\n{current}\n")

    with open(log_path, "w") as log:
        log.write(f"[0] SEED:\n{current}\n\n")

        for i in range(1, iterations + 1):
            # GPT2 recursively reflects on current text
            reflection_prompt = (
                f"Reflect on and zoom in on the following:\n\"{current.strip()}\"\n\nget closer:"
            )
            current = gpt_reflect(reflection_prompt)
            print(f"[{i}] TEXT:\n{current}\n")
            log.write(f"[{i}] TEXT:\n{current}\n\n")

            # Feed GPT2 output into image generation
            img_prompt = text_to_image_prompt(current)
            image = pipe(img_prompt, num_inference_steps=4, guidance_scale=0.0).images[0]

            img_path = os.path.join(output_dir, f"tame_{i:02d}.png")
            image.save(img_path)
            print(f"     → saved {img_path}\n")
            history.append({"iteration": i, "text": current, "image": img_path})

    print(f"Log saved to {log_path}")
    return history


if __name__ == "__main__":
    seed = "stronger prompt: Photorealistic 8K ultra sharp, cinematic bedroom lighting, Canon EOS R5 50mm f/1.8, shallow depth of field,sheen, masterpiece. (Young woman lounging back on crisp white sheets:1.4), (blue t-shirt completely long dark hair fanned out across pillow, sultry brown eyes locked on viewer with cheeky playful grin, one hand pointing directly at camera like 'your turn next':1.5), (perfect explicit :1.4),  Signature blue grippy socks with rubbery dotted soles hugging calves and feet, grippy . Background standard lamp, radio with soft green glow, fluorescent lights giving naughty warm-cyan ambiance. Large bubbly chaotic fun font title at very top exactly reading 'grippy socks- grippy box!'. Hyper-realistic, best quality, no artifacts, negative prompt: censored, blur, safe for work"
    fractal_dream(seed, iterations=5)