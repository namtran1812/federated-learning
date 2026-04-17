#!/usr/bin/env python3
"""
Interactive Federated LLM Token Generation
Real-time visualization of token generation from 3 federated clients
"""

import sys
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*70)
print("INTERACTIVE FEDERATED LLM - REAL-TIME TOKEN GENERATION")
print("="*70)

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    print("\n[Loading Model...]")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    print("✓ Model loaded!\n")
    
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)

num_clients = 3
temperature = 0.7

def aggregate_tokens(predictions):
    """Majority voting"""
    token_ids = [t[0] for t in predictions]
    return max(set(token_ids), key=token_ids.count)

while True:
    print("\n" + "="*70)
    user_prompt = input("Ask a question (or 'exit' to quit):\n> ").strip()
    
    if user_prompt.lower() == 'exit':
        print("\n✓ Goodbye!")
        break
    
    if not user_prompt:
        print("Please enter a question.")
        continue
    
    print("\n" + "-"*70)
    print("GENERATING TOKENS (3 Clients with Majority Voting)")
    print("-"*70 + "\n")
    
    input_ids = tokenizer.encode(user_prompt, return_tensors='pt')
    all_tokens = input_ids.clone()
    
    print(f"{'#':<4} {'Token':<15} {'ID':<8} {'Prob':<10} {'Clients Voted':<20}")
    print("-"*70)
    
    generated_text = user_prompt
    token_count = 0
    max_tokens = 20
    
    with torch.no_grad():
        for step in range(max_tokens):
            # Each client generates a prediction
            client_predictions = []
            
            for client_id in range(num_clients):
                outputs = model(all_tokens)
                logits = outputs.logits[0, -1, :]
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, 1)
                prob = probs[next_id].item()
                
                client_predictions.append((next_id.item(), prob, client_id + 1))
            
            # Combine via majority voting
            aggregated_id = aggregate_tokens(client_predictions)
            
            # Get probability for aggregated token
            outputs = model(all_tokens)
            logits = outputs.logits[0, -1, :]
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            agg_prob = probs[aggregated_id].item() if aggregated_id < len(probs) else 0.5
            
            token_text = tokenizer.decode([aggregated_id])
            generated_text += token_text
            token_count += 1
            
            # Show which clients voted for this token
            voting_clients = [f"C{p[2]}" for p in client_predictions if p[0] == aggregated_id]
            voted_str = ", ".join(voting_clients) if voting_clients else f"C1"
            
            print(f"{step+1:<4} {repr(token_text):<15} {aggregated_id:<8} {agg_prob:<10.4f} {voted_str:<20}")
            
            # Stop if we hit end token
            if token_text.strip() == '':
                if step > 5:
                    break
            
            all_tokens = torch.cat([all_tokens, torch.tensor([[aggregated_id]])], dim=1)
    
    print("\n" + "="*70)
    print("FINAL OUTPUT:")
    print("="*70)
    print(f"\n{generated_text}\n")
    print(f"✓ Generated {token_count} tokens with 3-client majority voting")

print("\n")
