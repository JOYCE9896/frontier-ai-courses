"""Use the trained model: look at its next-token probabilities, then write stories from different openings."""
import os
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from model import Config, TinyLlama

DATA = os.path.expanduser("~/.labs_data/tinystories/prepared")
RUN = os.path.expanduser("~/.labs_data/tinyllama_run")
dev = "mps"
tok = Tokenizer.from_file(os.path.join(DATA, "tokenizer.json"))
eot = tok.token_to_id("<|endoftext|>")
model = TinyLlama(Config()).to(dev)
model.load_state_dict(torch.load(os.path.join(RUN, "model.pt"), map_location=dev))
model.eval()

# 1. The model's guesses for the next token: the 5 most likely
print("=== top 5 next tokens ===")
for text in ["Once upon a", "Lily went to the park with her", "The cat was very", "Tom was sad because he lost his"]:
    ids = torch.tensor([tok.encode(text).ids], device=dev)
    with torch.no_grad():
        probs = F.softmax(model(ids)[0][0, -1], dim=-1)
    top = torch.topk(probs, 5)
    guesses = ", ".join(f"{tok.decode([i])!r} {p:.1%}" for p, i in zip(top.values.tolist(), top.indices.tolist()))
    print(f"{text} ___  ->  {guesses}")

# 2. Stories from different openings
def write(prompt, temperature, seed=0):
    torch.manual_seed(seed)
    idx = torch.tensor([tok.encode(prompt).ids], device=dev)
    out = model.generate(idx, 200, temperature=temperature, top_k=50, eot_id=eot)[0].tolist()
    return tok.decode([t for t in out if t != eot])

for prompt in ["Once upon a time, there was a little robot", "Sara and her dog went to the beach.", "The old tree in the forest"]:
    print(f"\n=== opening: {prompt} (temperature 0.8) ===")
    print(write(prompt, 0.8))

print("\n=== same opening, different temperatures ===")
for t in [0, 0.5, 1.0, 1.5]:
    print(f"\n--- temperature {t}{' (greedy: always the most likely token)' if t == 0 else ''} ---")
    print(write("One day, Ben found a box", t, seed=1))
