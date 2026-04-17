# Documentation Index

## Quick Links

### Getting Started
- **README.md** (root) - Project overview and quick start

### Core Documentation
1. **LLAMA_FEDERATED_APPROACH.md**
   - Detailed federated learning approach
   - Architecture diagrams
   - Expected results with Llama-3.1-8B
   - Implementation assumptions

2. **PHASE4_IMPROVEMENTS.md**
   - What changed from previous version
   - Problem statement and solution
   - Data structures
   - Expected improvements

3. **PRODUCTION_CHECKLIST.sh**
   - Readiness checklist
   - Expected results
   - Next steps for submission

### Scripts

#### Submission
- `scripts/submit_phase4.sh` - HiPerGator submission script

#### Testing
- `scripts/test_phase4_local.py` - Local verification with GPT-2
  - Tests federated learning mechanism
  - Verifies diversity metrics
  - Outputs sample results

## File Organization

```
docs/
├── README.md                      (This file)
├── LLAMA_FEDERATED_APPROACH.md   (Technical details)
├── PHASE4_IMPROVEMENTS.md         (Changes made)
└── PRODUCTION_CHECKLIST.sh        (Verification)
```

## Phase 4 Implementation Summary

### What It Does
- Uses Llama-3.1-8B language model
- 3 clients with different contexts
- Federated probability averaging
- 3 scenarios with 30 tokens each
- Tracks client preferences and diversity

### Key Features
- ✅ Independent context per client
- ✅ Federated averaging of probabilities
- ✅ Consensus token selection
- ✅ Client preference tracking
- ✅ Entropy calculations
- ✅ Step-by-step metrics

### Expected Results
- Diversity Rate: 65-80%
- Execution Time: 20-30 minutes
- Output Format: JSON with detailed metrics

## How to Use

### 1. Local Testing
```bash
python scripts/test_phase4_local.py
```
This uses GPT-2 to test the mechanism locally.

### 2. HiPerGator Submission
See main README.md for detailed steps.

### 3. Monitoring
After submission, check:
```bash
squeue -u nam.tran1
```

### 4. Results
Results will be in:
```
federated-llm/results/phase_4_generations.json
```

## Understanding the Output

### JSON Structure
```json
{
  "timestamp": "2026-04-17T14:32:15",
  "model": "Llama-3.1-8B-Instruct",
  "scenarios": [
    {
      "scenario_name": "AI Definition",
      "result": {
        "client_outputs": [
          {
            "client_id": 1,
            "context_prefix": "Client 1 (ML Engineer)...",
            "output": "...generated text..."
          }
        ],
        "steps": [
          {
            "selected_token": " systems",
            "client_preferences": [
              {"client_id": 1, "preferred_token": " systems"},
              {"client_id": 2, "preferred_token": " consciousness"},
              {"client_id": 3, "preferred_token": " systems"}
            ]
          }
        ]
      }
    }
  ]
}
```

### Key Fields
- **client_outputs**: Final generated text per client
- **client_preferences**: What each client preferred at each step
- **aggregated_probability**: Probability of selected token
- **entropy**: Diversity measure

## Troubleshooting

### Local Testing Issues
```bash
# Make sure GPT-2 is installed
pip install transformers torch

# Run with verbose output
python scripts/test_phase4_local.py
```

### HiPerGator Submission Issues
1. Check VPN connection
2. Verify HF token is set: `echo $HF_TOKEN`
3. Check SLURM logs: `tail -f slurm-*.out`

### File Organization Issues
All paths assume root directory is `/Users/namtran/LLM`

## References

See individual documentation files for:
- Detailed technical explanation: `LLAMA_FEDERATED_APPROACH.md`
- Implementation changes: `PHASE4_IMPROVEMENTS.md`
- Verification checklist: `PRODUCTION_CHECKLIST.sh`

---

**Last Updated**: April 17, 2026
**Status**: Production Ready ✅
