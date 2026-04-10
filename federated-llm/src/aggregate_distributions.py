#!/usr/bin/env python3
"""
PHASE 3: Average the Distributions Together

Key math: If we average 3 probability distributions, do we still get a valid distribution?
Answer: Yes! Average of valid distributions = valid distribution.
Result: Federated approach is mathematically sound.
"""

import warnings, json, torch, os
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime

warnings.filterwarnings('ignore')
print("\nPHASE 3: Distribution Aggregation\n" + "="*60)

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL).to(DEVICE)
    print("✓ Model loaded\n")
except Exception as e:
    print(f"❌ {e}")
    exit(1)

def get_dist(prompt):
    """Get next-token probability distribution for a prompt"""
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output = model(inputs["input_ids"])
    return torch.softmax(output.logits[0, -1, :], dim=-1)

os.makedirs("results", exist_ok=True)
all_results = {"timestamp": datetime.now().isoformat(), "scenarios": []}

scenarios = [
    {
        "name": "AI Perspectives",
        "question": "What is artificial intelligence?",
        "contexts": [
            "I'm a machine learning engineer.",
            "I'm a philosopher interested in AI ethics.",
            "I'm a neuroscientist studying intelligence."
        ]
    },
    {
        "name": "Privacy Perspectives", 
        "question": "How important is data privacy?",
        "contexts": [
            "I work on privacy-preserving tech.",
            "I focus on regulatory compliance.",
            "I research formal privacy guarantees."
        ]
    }
]

for scenario in scenarios:
    print(f"Scenario: {scenario['name']}")
    print(f"Question: {scenario['question']}\n")
    
    result = {"name": scenario["name"], "question": scenario["question"]}
    
    # Get each client's distribution
    dists = []
    for i, context in enumerate(scenario["contexts"], 1):
        prompt = f"{context}\n{scenario['question']}"
        dist = get_dist(prompt)
        dists.append(dist)
        
        best_token = tokenizer.decode([torch.argmax(dist).item()])
        prob = torch.max(dist).item()
        print(f"  Client {i}: '{best_token}' ({prob:.1%})")
    
    # Average them (federated approach)
    federated_dist = torch.mean(torch.stack(dists), dim=0)
    
    # Check: sum to 1.0? (validity check)
    dist_sum = torch.sum(federated_dist).item()
    
    # What do we get?
    fed_token = tokenizer.decode([torch.argmax(federated_dist).item()])
    fed_prob = torch.max(federated_dist).item()
    
    print(f"\n  Federated (average): '{fed_token}' ({fed_prob:.1%})")
    print(f"  Distribution sums to: {dist_sum:.6f} ✓ (should be 1.0)")
    
    result["federated_token"] = fed_token
    result["federated_prob"] = fed_prob
    result["valid"] = abs(dist_sum - 1.0) < 0.001
    
    all_results["scenarios"].append(result)
    print()

with open("results/phase_3_aggregation.json", "w") as f:
    json.dump(all_results, f)

print(f"{'='*60}\n✓ Saved to results/phase_3_aggregation.json")
print("\nKey insight: Averaging distributions works mathematically!")
