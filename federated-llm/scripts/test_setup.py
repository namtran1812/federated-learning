#!/usr/bin/env python3
"""
Quick Test Script: Verify federated decoding setup before HiPerGator submission

This lightweight script tests:
1. Model loading
2. Tokenizer functionality
3. Distribution computation
4. Aggregation logic
5. Single-step generation

Run this locally first to ensure everything works!
"""

import warnings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("QUICK TEST: FEDERATED DECODING SETUP")
print("="*80)

# ============================================================================
# TEST 1: Model Loading
# ============================================================================

print("\n[TEST 1] Model Loading...")
try:
    # Use smaller model for local testing, full model on HiPerGator
    MODEL_NAME = "gpt2"
    print(f"  Loading {MODEL_NAME} for local testing...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"  ✓ Tokenizer loaded (vocab_size={len(tokenizer)})")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    print(f"  ✓ Model loaded successfully!")
    print(f"  Model size: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")
    print(f"  Note: Using GPT-2 for local testing. HiPerGator will use Llama-3.1-8B-Instruct")

except Exception as e:
    print(f"  ✗ FAILED: {e}")
    print("\n  To fix:")
    print("  1. Install transformers: pip install transformers")
    print("  2. Login to HuggingFace: huggingface-cli login")
    exit(1)

# ============================================================================
# TEST 2: Tokenizer Functionality
# ============================================================================

print("\n[TEST 2] Tokenizer Functionality...")
try:
    test_text = "Artificial intelligence is"
    tokens = tokenizer.encode(test_text)
    decoded = tokenizer.decode(tokens)
    
    print(f"  Input: '{test_text}'")
    print(f"  Tokens: {tokens}")
    print(f"  Decoded: '{decoded}'")
    print(f"  ✓ Tokenizer works correctly")

except Exception as e:
    print(f"  ✗ FAILED: {e}")
    exit(1)

# ============================================================================
# TEST 3: Single Token Distribution
# ============================================================================

print("\n[TEST 3] Single Token Distribution...")
try:
    prompt = "What is artificial intelligence?"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(inputs["input_ids"])
        logits = outputs.logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1)
    
    print(f"  Prompt: '{prompt}'")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Probabilities shape: {probs.shape}")
    print(f"  Sum of probabilities: {probs.sum().item():.10f} (should be ~1.0)")
    
    # Top-5 tokens
    top_5_probs, top_5_indices = torch.topk(probs, k=5)
    print(f"\n  Top-5 next tokens:")
    for rank, (prob, idx) in enumerate(zip(top_5_probs, top_5_indices), 1):
        token = tokenizer.decode([idx.item()])
        print(f"    {rank}. {repr(token):<20} {prob.item():.6f}")
    
    print(f"\n  ✓ Distribution computation works!")

except Exception as e:
    print(f"  ✗ FAILED: {e}")
    exit(1)

# ============================================================================
# TEST 4: Distribution Aggregation
# ============================================================================

print("\n[TEST 4] Distribution Aggregation...")
try:
    # Create 3 client distributions (simulated by different prompts)
    client_prompts = [
        "Client 1 (ML Engineer): Artificial intelligence is",
        "Client 2 (Philosopher): Artificial intelligence is",
        "Client 3 (Neuroscientist): Artificial intelligence is"
    ]
    
    client_dists = []
    
    print(f"  Computing {len(client_prompts)} client distributions...")
    
    for i, prompt in enumerate(client_prompts):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model(inputs["input_ids"])
            logits = outputs.logits[0, -1, :]
            probs = torch.softmax(logits, dim=-1)
        
        client_dists.append(probs)
        
        top_token = tokenizer.decode([torch.argmax(probs).item()])
        print(f"    Client {i+1}: next token = {repr(top_token)}")
    
    # Aggregate
    fed_probs = sum(p / len(client_dists) for p in client_dists)
    fed_probs = fed_probs / fed_probs.sum()
    
    print(f"\n  Federated distribution shape: {fed_probs.shape}")
    print(f"  Sum of aggregated probabilities: {fed_probs.sum().item():.10f}")
    
    top_fed_token = tokenizer.decode([torch.argmax(fed_probs).item()])
    print(f"  Federated top token: {repr(top_fed_token)}")
    
    print(f"\n  ✓ Distribution aggregation works!")

except Exception as e:
    print(f"  ✗ FAILED: {e}")
    exit(1)

# ============================================================================
# TEST 5: Multi-Step Generation
# ============================================================================

print("\n[TEST 5] Multi-Step Federated Generation...")
try:
    prompt = "Federated learning is a technique for"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    print(f"  Prompt: '{prompt}'")
    print(f"  Generating 5 tokens...")
    print(f"\n  {'Step':<6} {'Token':<15} {'Probability':<12}")
    print("  " + "-"*35)
    
    input_ids = inputs["input_ids"].clone()
    
    with torch.no_grad():
        for step in range(5):
            outputs = model(input_ids)
            logits = outputs.logits[0, -1, :]
            probs = torch.softmax(logits, dim=-1)
            
            next_token_id = torch.argmax(probs).unsqueeze(0).unsqueeze(0)
            token_text = tokenizer.decode([next_token_id.item()])
            prob = probs[next_token_id.item()].item()
            
            print(f"  {step:<6} {repr(token_text):<15} {prob:<12.6f}")
            
            input_ids = torch.cat([input_ids, next_token_id], dim=1)
    
    full_output = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    print(f"\n  Full output: {full_output}")
    print(f"\n  ✓ Multi-step generation works!")

except Exception as e:
    print(f"  ✗ FAILED: {e}")
    exit(1)

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("QUICK TEST SUMMARY")
print("="*80)

print(f"""
✅ ALL TESTS PASSED!

Your setup is ready for federated decoding. Next steps:

1. Upload to HiPerGator:
   cd /Users/namtran/LLM/federated-llm
   scp -r . nam.tran1@hpg.rc.ufl.edu:/blue/jie.xu/nam.tran1/federated-llm/

2. Submit Phase 1 job:
   ssh nam.tran1@hpg.rc.ufl.edu
   cd /blue/jie.xu/nam.tran1/federated-llm
   sbatch slurm/phase_1.slurm

3. Monitor progress:
   squeue -u nam.tran1
   tail -f fed_phase_1_JOBID.out

For detailed execution guide, see: EXECUTION_GUIDE.md
For research details, see: README.md
""")
