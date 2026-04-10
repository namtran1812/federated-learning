# Federated LLM Decoding
## Token-Level Distribution Aggregation with Llama-3.1-8B

### The Big Idea
What if multiple AI "agents" with different knowledge backgrounds work together to generate better text? Each agent gets the same prompt but with different context, produces their own next-token probabilities, and we average them together. Does this produce text that's smarter than any single agent?

**Formula:** $p_{\text{fed}}(v) = \frac{1}{M}\sum_{i=1}^{M} p_i(v)$ 
*(Each client's distribution gets equal weight, then we sample from the average)*

### Quick Deploy
```bash
bash submit_to_hipergator.sh
```
This uploads everything and submits Phase 1 to HiPerGator.

### What You Get
- **Phase 1** (~10 min): Show what tokens the model likes at each generation step
- **Phase 2** (~20 min): Prove that different contexts → different token choices  
- **Phase 3** (~15 min): Average distributions together and check they're still valid
- **Phase 4** (~30 min): Generate complete text using the federated approach

**Total runtime:** ~90 minutes on GPU

### Files You'll See
```
1_extract_token_scores.py          ← Run first: what tokens does model like?
2_compare_client_contexts.py       ← Do different perspectives matter?
3_aggregate_distributions.py       ← Average the distributions together
4_generate_federated_text.py       ← Generate full text using federation
test_setup.py                      ← Check everything works locally
submit_to_hipergator.sh            ← Upload and submit to HiPerGator
DEPLOY_TO_HIPERGATOR.md            ← Detailed deployment guide
slurm/                             ← Job scripts for HiPerGator
results/                           ← Where outputs go
```

## Mathematical Foundations

### Comparability (Phase 1)
All clients use the same model and tokenizer, so:
$$p_i^{(t)} \in \Delta_{|V|-1}$$

Each distribution lies in the same probability simplex over vocabulary $V$.

### Aggregation (Phase 3)
A weighted average of probability distributions is still a valid distribution:
$$p_{\text{fed}}^{(t)} = \sum_i w_i p_i^{(t)} \in \Delta_{|V|-1}$$

### Decoding (Phase 4)
At each step, sample or select from the aggregated distribution:
$$x_t = \arg\max_v p_{\text{fed}}^{(t)}(v)$$

## Key Metrics

1. **Validity**: Complete sentence output
2. **Source Coverage**: Information from each client included
3. **Fact Coverage**: Information from centralized context preserved

Target relationship:
$$\text{Cov}(\text{Local}) \leq \text{Cov}(\text{Federated}) \leq \text{Cov}(\text{Centralized})$$

## HiPerGator Setup

Load required modules:
```bash
module load gcc/11.2.0 python/3.11 cuda/12.2
pip install torch transformers accelerate bitsandbytes
```

## Author Notes

This implements the token-level aggregation approach described in the research question, with reference to Fed-ICL (ICML 2025) as a theoretical comparison point for understanding how local context acts as an implicit update to model behavior.
