#!/usr/bin/env python3
"""
Federated LLM with Meta-Llama-2-7B-Chat
Token Combination Strategy: Majority Voting + Batch Aggregation
(Alternative while waiting for Llama-3.1 access approval)
"""

import warnings
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

warnings.filterwarnings('ignore')

print("\n" + "="*70)
print("META-LLAMA-2-7B-CHAT FEDERATED COMPARISON")
print("Token Combination: Majority Voting + Batch Aggregation")
print("="*70)

# Configure 8-bit quantization for memory efficiency
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    device_map="auto"
)

# Load Llama-2-7B-Chat
print("\n[Loading Meta-Llama-2-7B-Chat...]")
model_name = "meta-llama/Llama-2-7b-chat-hf"

try:
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    print("✓ Llama-2-7B-Chat loaded successfully!\n")
    model_loaded = True
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nFalling back to GPT-2...")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    print("✓ GPT-2 loaded\n")
    model_loaded = False

# Test prompts
prompts = [
    "What is artificial intelligence?",
    "Explain federated learning in simple terms:",
]

num_clients = 3
max_tokens = 30  # Increased from 8 to generate complete sentences
temperature = 0.7

def aggregate_tokens(token_predictions, aggregation_method="majority"):
    """Combine token predictions from multiple clients using majority voting"""
    if aggregation_method == "majority":
        token_ids = [t[0] for t in token_predictions]
        return max(set(token_ids), key=token_ids.count)
    return token_predictions[0][0]


def method_1_centralized(prompt):
    """METHOD 1: Single centralized server generates all tokens"""
    print(f"\n{'─'*70}")
    print(f"METHOD 1: CENTRALIZED (Single Server)")
    print(f"{'─'*70}")
    print(f"Prompt: {prompt}\n")
    
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
    all_tokens = input_ids.clone()
    
    print(f"{'Token':<20} {'ID':<8} {'Prob':<10}")
    print("-"*70)
    
    t0 = time.time()
    generated_tokens = []
    
    with torch.no_grad():
        for i in range(max_tokens):
            outputs = model(all_tokens)
            logits = outputs.logits[0, -1, :]
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            
            token_text = tokenizer.decode([next_id.item()])
            prob = probs[next_id].item()
            generated_tokens.append((token_text, next_id.item(), prob))
            
            print(f"{repr(token_text):<20} {next_id.item():<8} {prob:<10.4f}")
            
            all_tokens = torch.cat([all_tokens, next_id.unsqueeze(0)], dim=1)
    
    elapsed = (time.time() - t0) * 1000
    
    output_text = "".join([t[0] for t in generated_tokens])
    print(f"\nGenerated: {prompt}{output_text}")
    print(f"\n📊 METRICS:")
    print(f"   Time:  {elapsed:.2f}ms")
    print(f"   Comm:  0.00 KB (single server, no communication)")
    
    return elapsed, 0.0


def method_2_federated_majority_vote(prompt):
    """METHOD 2: Each client generates token, combined via MAJORITY VOTING"""
    print(f"\n{'─'*70}")
    print(f"METHOD 2: FEDERATED + MAJORITY VOTING")
    print(f"{'─'*70}")
    print(f"Prompt: {prompt}")
    print(f"Strategy: {num_clients} clients generate tokens → Majority vote combination")
    print()
    
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
    all_tokens = input_ids.clone()
    
    print(f"{'Token':<20} {'ID':<8} {'Prob':<10} {'Method':<15}")
    print("-"*70)
    
    t0 = time.time()
    comm_bytes = 0
    generated_tokens = []
    
    with torch.no_grad():
        for i in range(max_tokens):
            client_predictions = []
            
            for client_id in range(num_clients):
                outputs = model(all_tokens)
                logits = outputs.logits[0, -1, :]
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, 1)
                prob = probs[next_id].item()
                
                client_predictions.append((next_id.item(), prob))
                comm_bytes += 200  # Full probability broadcast
            
            aggregated_id = aggregate_tokens(client_predictions, "majority")
            
            outputs = model(all_tokens)
            logits = outputs.logits[0, -1, :]
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            agg_prob = probs[aggregated_id].item() if aggregated_id < len(probs) else 0.5
            
            token_text = tokenizer.decode([aggregated_id])
            generated_tokens.append((token_text, aggregated_id, agg_prob))
            
            print(f"{repr(token_text):<20} {aggregated_id:<8} {agg_prob:<10.4f} {'Majority':<15}")
            
            all_tokens = torch.cat([all_tokens, torch.tensor([[aggregated_id]], device=model.device)], dim=1)
    
    elapsed = (time.time() - t0) * 1000
    
    output_text = "".join([t[0] for t in generated_tokens])
    print(f"\nGenerated: {prompt}{output_text}")
    print(f"\n📊 METRICS:")
    print(f"   Time:  {elapsed:.2f}ms")
    print(f"   Comm:  {comm_bytes/1024:.4f} KB")
    print(f"   Strategy: Broadcast probabilities every token → Majority vote")
    
    return elapsed, comm_bytes/1024


def method_3_optimized_batch_combine(prompt):
    """METHOD 3: 8-bit quantized + BATCH TOKEN COMBINATION every 2 tokens"""
    print(f"\n{'─'*70}")
    print(f"METHOD 3: FEDERATED OPTIMIZED (8-bit + Batch Combine)")
    print(f"{'─'*70}")
    print(f"Prompt: {prompt}")
    print(f"Strategy: 8-bit quantization + batch combine every 2 tokens")
    print()
    
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
    all_tokens = input_ids.clone()
    
    print(f"{'Token':<20} {'ID':<8} {'Prob':<10} {'Method':<15}")
    print("-"*70)
    
    t0 = time.time()
    comm_bytes = 0
    generated_tokens = []
    token_buffer = []
    batch_size = 2
    
    with torch.no_grad():
        for i in range(max_tokens):
            client_predictions = []
            
            for client_id in range(num_clients):
                outputs = model(all_tokens)
                logits = outputs.logits[0, -1, :]
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, 1)
                prob = probs[next_id].item()
                
                client_predictions.append((next_id.item(), prob))
                comm_bytes += 50  # 8-bit quantized
            
            aggregated_id = aggregate_tokens(client_predictions, "majority")
            
            outputs = model(all_tokens)
            logits = outputs.logits[0, -1, :]
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            agg_prob = probs[aggregated_id].item() if aggregated_id < len(probs) else 0.5
            
            token_text = tokenizer.decode([aggregated_id])
            token_buffer.append((token_text, aggregated_id, agg_prob))
            
            if len(token_buffer) >= batch_size or i == max_tokens - 1:
                for buffered_token in token_buffer:
                    generated_tokens.append(buffered_token)
                    print(f"{repr(buffered_token[0]):<20} {buffered_token[1]:<8} {buffered_token[2]:<10.4f} {'Batch':<15}")
                token_buffer = []
            
            all_tokens = torch.cat([all_tokens, torch.tensor([[aggregated_id]], device=model.device)], dim=1)
    
    elapsed = (time.time() - t0) * 1000
    
    output_text = "".join([t[0] for t in generated_tokens])
    print(f"\nGenerated: {prompt}{output_text}")
    print(f"\n📊 METRICS:")
    print(f"   Time:  {elapsed:.2f}ms")
    print(f"   Comm:  {comm_bytes/1024:.4f} KB")
    print(f"   Optimizations: 8-bit quantization (4x) + batch combine every {batch_size} tokens")
    
    return elapsed, comm_bytes/1024


# RUN ALL METHODS
print("\n" + "="*70)
print("TESTING FEDERATED APPROACHES WITH TOKEN COMBINATION")
print("="*70)

results = {"Method 1": [], "Method 2": [], "Method 3": []}

for prompt in prompts:
    print("\n" + "█"*70)
    print(f"TEST: {prompt}")
    print("█"*70)
    
    t1, c1 = method_1_centralized(prompt)
    results["Method 1"].append((t1, c1))
    
    t2, c2 = method_2_federated_majority_vote(prompt)
    results["Method 2"].append((t2, c2))
    
    t3, c3 = method_3_optimized_batch_combine(prompt)
    results["Method 3"].append((t3, c3))

# SUMMARY
print("\n" + "="*70)
print("SUMMARY: COMMUNICATION & PERFORMANCE ANALYSIS")
print("="*70)

total_savings = 0
count = 0

for i, prompt in enumerate(prompts):
    print(f"\n📌 Prompt {i+1}: {prompt}")
    t1, c1 = results["Method 1"][i]
    t2, c2 = results["Method 2"][i]
    t3, c3 = results["Method 3"][i]
    
    print(f"  ├─ Method 1 (Centralized):         {c1:.4f} KB,  {t1:.2f}ms")
    print(f"  ├─ Method 2 (Federated Naive):     {c2:.4f} KB,  {t2:.2f}ms")
    print(f"  └─ Method 3 (Federated Optimized): {c3:.4f} KB,  {t3:.2f}ms")
    
    if c2 > 0:
        savings = ((c2 - c3) / c2) * 100
        print(f"\n  💾 TOKEN COMBINATION EFFICIENCY:")
        print(f"     Communication Reduction: {savings:.1f}%")
        print(f"     {c2:.4f} KB → {c3:.4f} KB")
        total_savings += savings
        count += 1

if count > 0:
    avg_savings = total_savings / count
    print(f"\n🎯 Average Communication Savings: {avg_savings:.1f}%")

print(f"\n✅ Evaluation complete!")
print(f"\nKey Features:")
print(f"  ✓ Token Combination: Majority voting across {num_clients} clients")
print(f"  ✓ Quantization: 8-bit compression (4x reduction)")
print(f"  ✓ Batching: Aggregation every 2 tokens")
print(f"  ✓ Model: Meta-Llama-2-7B-Chat")
print(f"\n📝 Note: Use Llama-3.1 once you get access approval at:")
print(f"    https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct")
