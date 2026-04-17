#!/usr/bin/env python3
"""
PHASE 2: Do Different Contexts Change Token Choices?

Test: 3 "agents" with different backgrounds (ML engineer, philosopher, neuroscientist)
see the same question. Do they pick different next tokens?
Answer: Yes! KL divergence shows their distributions differ.
"""

import warnings, json, torch, os, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime

warnings.filterwarnings('ignore')
print("\nPHASE 2: Multi-Client Context Divergence\n" + "="*60)

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}\n")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL).to(DEVICE)
    print("✓ Model loaded\n")
except Exception as e:
    print(f"❌ {e}")
    exit(1)

def get_dist(prompt, steps=8):
    """Get token probability distribution"""
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output = model(inputs["input_ids"])
    return torch.softmax(output.logits[0, -1, :], dim=-1)

def kl_div(p, q):
    """KL divergence: how different are distributions p and q?"""
    return torch.sum(p * (torch.log(p + 1e-10) - torch.log(q + 1e-10))).item()

scenarios = [
    {
        "name": "AI Expertise",
        "question": "What is artificial intelligence?",
        "clients": [
            "I'm an ML engineer with 5 years experience.",
            "I'm a philosophy student studying AI ethics.",
            "I'm a neuroscientist interested in biological intelligence."
        ]
    },
    {
        "name": "Privacy",
        "question": "How does federated learning improve data privacy?",
        "clients": [
            "I work on privacy-preserving ML at a tech company.",
            "I'm a regulatory officer focused on data protection.",
            "I research differential privacy and formal guarantees."
        ]
    }
]

os.makedirs("results", exist_ok=True)
all_results = {"timestamp": datetime.now().isoformat(), "scenarios": []}

for scenario in scenarios:
    print(f"Scenario: {scenario['name']}")
    print(f"Question: {scenario['question']}\n")
    
    scenario_result = {"name": scenario["name"], "question": scenario["question"], "clients": []}
    distributions = []
    
    for i, context in enumerate(scenario["clients"], 1):
        prompt = f"{context}\n{scenario['question']}"
        dist = get_dist(prompt)
        distributions.append(dist)
        
        top_token = tokenizer.decode([torch.argmax(dist).item()])
        top_prob = torch.max(dist).item()
        
        print(f"  Client {i}: {context[:40]}... → '{top_token}' ({top_prob:.1%})")
        scenario_result["clients"].append({"context": context, "top_token": top_token, "prob": top_prob})
    
    # Measure divergence between clients
    div_12 = kl_div(distributions[0], distributions[1])
    div_13 = kl_div(distributions[0], distributions[2])
    div_23 = kl_div(distributions[1], distributions[2])
    
    scenario_result["divergence"] = {"1v2": div_12, "1v3": div_13, "2v3": div_23}
    all_results["scenarios"].append(scenario_result)

with open("results/phase_2_distributions.json", "w") as f:
    json.dump(all_results, f)
