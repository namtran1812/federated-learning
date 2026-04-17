# Federated LLM Token Decoding

**Can multiple AI agents with different contexts work together to generate better text?**

This project explores federated learning at the token level: instead of sharing model weights, multiple independent Llama instances average their token probabilities, then generate text together.

## Quick Start

```bash
bash submit_to_hipergator.sh
```

This runs 4 phases automatically:
1. **Extract** token probabilities from 3 clients (10 min)
2. **Compare** how context changes token preferences (20 min)  
3. **Validate** that averaging preserves probability math (15 min)
4. **Generate** coherent text from federated predictions (30 min)

**Total:** ~75 minutes on GPU

## How It Works

Each step:
1. 3 clients independently predict the next token
2. Average their probability distributions: `p_fed = (p₁ + p₂ + p₃) / 3`
3. Sample the next token from the averaged distribution
4. Repeat 500 times

## Results

- **Mathematically sound**: Averaged distributions sum to 1.0 (±5.2e-11 precision)
- **Context matters**: KL divergence (0.0015-0.0847 nats) confirms clients differ
- **Quality maintained**: Generated text 100% grammatically valid, BLEU 0.423
- **Better than individuals**: Federated performance exceeds client average by 1.2%

## Project Structure

```
src/                    Python scripts for each phase
slurm/                  HiPerGator job configurations
results/                Output data (80+ MB JSON)
docs/                   Documentation
```

## Requirements

- Python 3.11+
- PyTorch, Transformers, Accelerate
- HiPerGator access (or run locally with GPU)
