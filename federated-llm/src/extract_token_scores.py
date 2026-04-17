#!/usr/bin/env python3
"""
PHASE 1: What Tokens Does the Model Like?

Simple idea: Feed prompts to the model and ask "what token comes next?"
Show the top 10 possibilities with confidence scores.
"""

import warnings, json, torch, os
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime

warnings.filterwarnings('ignore')
print("\nPHASE 1: Token Distribution Extraction\n" + "="*60)

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE} | Model: {MODEL}")

try:
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL).to(DEVICE)
    print("✓ Ready\n")
except Exception as e:
    print(f"❌ {e}\nRun: huggingface-cli login")
    exit(1)

def extract_tokens(prompt, steps=15):
    """Generate text step-by-step, recording token probabilities"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    
    print(f"Prompt: {prompt}")
    print(f"Length: {input_ids.shape[1]} tokens\n")
    
    results = {"prompt": prompt, "steps": []}
    
    with torch.no_grad():
        for i in range(steps):
            out = model(input_ids)
            probs = torch.softmax(out.logits[0, -1, :], dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
            
            top_probs, top_ids = torch.topk(probs, 10)
            top_tokens = [tokenizer.decode([id_.item()]) for id_ in top_ids]
            
            print(f"  Step {i}: '{top_tokens[0]}' ({top_probs[0]:.1%})")
            
            results["steps"].append({
                "token": top_tokens[0],
                "prob": float(top_probs[0]),
                "entropy": entropy,
                "top_10": [{"text": t, "prob": float(p)} for t, p in zip(top_tokens, top_probs)]
            })
            
            input_ids = torch.cat([input_ids, torch.argmax(probs).unsqueeze(0).unsqueeze(0)], dim=1)
    
    results["output"] = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return results

os.makedirs("results", exist_ok=True)
all_results = {"timestamp": datetime.now().isoformat(), "prompts": []}

for prompt in ["What is federated learning?", "Explain AI in one sentence.", "How does token decoding work?"]:
    print(f"\n{'-'*60}\n")
    all_results["prompts"].append(extract_tokens(prompt))

with open("results/phase_1_scores.json", "w") as f:
    json.dump(all_results, f)
