import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from huggingface_hub import login

<<<<<<< HEAD
=======
HF_TOKEN = "REMOVED_HF_TOKEN"
>>>>>>> parent of 5645530 (Update fsmts.py)
login(token=HF_TOKEN)

model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model.eval()

def generate(prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=1.0,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def extract_transform(before, after):
    """
    Naive transform: get the added suffix from model's response.
    """
    if after.startswith(before):
        return after[len(before):].strip()
    return after

def fractal_dream(seed, iterations=10):
    current = seed
    history = [seed]
    print(f"[0] {seed}")

    for i in range(1, iterations + 1):
        full = generate(current)
        delta = extract_transform(current, full)
        current += " " + delta
        history.append(current)
        print(f"[{i}] {delta}")

    return history

if __name__ == "__main__":
    seed = "Feed the model a seed sequence S₀ Generate a continuation R₀ Extract the appended content T₀ = Δ(S₀, R₀) Form the new input S₁ = S₀ + T₀"
    fractal_dream(seed, iterations=10)
