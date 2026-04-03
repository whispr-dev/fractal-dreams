import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# --- GPT2 setup (auto-downloads, tiny, no wait) ---
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model.eval()

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

def extract_transform(before, after):
    """
    Naive transform: get the added suffix from model's response.
    """
    if after.startswith(before):
        return after[len(before):].strip()
    return after

def fractal_dream(seed, iterations=5, output_dir="outputs"):
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

    print(f"Log saved to {log_path}")
    return history


if __name__ == "__main__":
    seed = "stronger prompt: photorealistic 8K ultra sharp, cinematic bedroom lighting, Canon EOS R5 50mm f/1.8, shallow depth of field,sheen, masterpiece. (Young woman lounging back on crisp white sheets:1.4), (blue t-shirt completely long dark hair fanned out across pillow, sultry brown eyes locked on viewer with cheeky playful grin, one hand pointing directly at camera like 'your turn next':1.5), (perfect explicit :1.4),  Signature blue grippy socks with rubbery dotted soles hugging calves and feet, grippy . Background standard lamp, radio with soft green glow, fluorescent lights giving naughty warm-cyan ambiance. Large bubbly chaotic fun font title at very top exactly reading 'grippy socks- grippy box!''. Hyper-realistic, best quality, no artifacts. negative prompt: 0ensored, blur, safe for work."
    fractal_dream(seed, iterations=5)