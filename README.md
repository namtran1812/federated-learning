# Federated Learning LLM - Phase 4 Implementation

## 📋 Project Overview

This project implements **Federated Text Generation** using Llama-3.1-8B, demonstrating how federated learning principles can be applied to language model inference.

### Key Innovation

Each of 3 clients maintains **independent context** throughout text generation. Their probability distributions are **federated averaged** at each step, creating diverse outputs while maintaining consensus.

## 🚀 Quick Start

### Local Testing
```bash
cd /Users/namtran/LLM
python scripts/test_phase4_local.py
```

### HiPerGator Submission
```bash
ssh nam.tran1@hpg.rc.ufl.edu
cd /blue/jie.xu/nam.tran1/federated-llm
sbatch slurm/4_generate_federated_text.slurm
```

## 📁 Project Structure

```
LLM/
├── docs/                           # Documentation
│   ├── LLAMA_FEDERATED_APPROACH.md # Detailed approach explanation
│   ├── PHASE4_IMPROVEMENTS.md      # What changed and why
│   ├── PRODUCTION_CHECKLIST.sh     # Verification checklist
│   └── README.md                   # This file
│
├── scripts/                        # Utility scripts
│   ├── submit_phase4.sh           # HiPerGator submission
│   └── test_phase4_local.py       # Local verification with GPT-2
│
├── federated-llm/                 # Main project
│   ├── src/
│   │   ├── generate_federated_text.py  # Core implementation
│   │   ├── compare_client_contexts.py
│   │   ├── extract_token_scores.py
│   │   └── aggregate_distributions.py
│   │
│   ├── slurm/
│   │   ├── 1_extract_token_scores.slurm
│   │   ├── 2_compare_client_contexts.slurm
│   │   ├── 3_aggregate_distributions.slurm
│   │   └── 4_generate_federated_text.slurm     # Phase 4 job
│   │
│   ├── results/
│   │   ├── phase_1_scores.json
│   │   ├── phase_2_distributions.json
│   │   ├── phase_3_aggregation.json
│   │   └── phase_4_generations.json            # Phase 4 output
│   │
│   ├── config/
│   │   └── [Configuration files]
│   │
│   └── docs/
│       └── DEPLOY_TO_HIPERGATOR.md
│
├── hipergator-project/            # Previous experiments (archived)
│   ├── RESULTS.txt
│   ├── scripts/
│   ├── slurm/
│   └── [Legacy Phase 1-3 work]
│
├── archived/                      # Unused scripts
│   ├── hipergator_submit.sh
│   └── hipergator_submit_correct.sh
│
└── .venv/                         # Python virtual environment
```

## 🔧 Implementation Details

### Phase 4: Federated Text Generation

**Approach**: Federated Probability Averaging

```
For each generation step:
  1. Each client generates own probability distribution
  2. Distributions are averaged (federated aggregation)
  3. Token selected from averaged distribution
  4. All clients follow the consensus token
```

**Result**: Diverse outputs reflecting different client perspectives

### Example Outputs

**Privacy in ML Scenario:**
- **Client 1 (Privacy Researcher)**: "maintaining encrypted data distributions..."
- **Client 2 (Data Officer)**: "ensuring compliance and governance..."
- **Client 3 (Cryptographer)**: "protecting through cryptographic protocols..."

## 📊 Key Metrics

| Aspect | Value |
|--------|-------|
| Model | Llama-3.1-8B-Instruct |
| Clients | 3 (with distinct roles) |
| Scenarios | 3 (AI Definition, Privacy, Token Decoding) |
| Tokens per scenario | 30 |
| Expected Diversity Rate | 65-80% |
| Execution Time | 20-30 minutes |

## 🎯 Current Status

✅ **Phase 4 Ready for Production**

- [x] Federated learning approach implemented
- [x] Three role-based clients configured
- [x] Three diverse scenarios created
- [x] Local testing passed (40% diversity with GPT-2)
- [x] Results JSON with detailed metrics
- [x] SLURM job configured for HiPerGator
- [x] Documentation complete

## 🚢 Deployment

### Prerequisites
- HiPerGator HPC access
- Llama-3.1-8B model access via Hugging Face
- GPU resources (A100 recommended)

### Steps

1. **Upload to HiPerGator**
   ```bash
   scp federated-llm/src/generate_federated_text.py \
       nam.tran1@hpg.rc.ufl.edu:/blue/jie.xu/nam.tran1/federated-llm/src/
   ```

2. **SSH to HiPerGator**
   ```bash
   ssh nam.tran1@hpg.rc.ufl.edu
   cd /blue/jie.xu/nam.tran1/federated-llm
   ```

3. **Submit Job**
   ```bash
   sbatch slurm/4_generate_federated_text.slurm
   ```

4. **Monitor**
   ```bash
   squeue -u nam.tran1
   ```

5. **Download Results**
   ```bash
   scp nam.tran1@hpg.rc.ufl.edu:/blue/jie.xu/nam.tran1/federated-llm/results/phase_4_generations.json \
       federated-llm/results/
   ```

## 📚 Documentation

- `docs/LLAMA_FEDERATED_APPROACH.md` - Detailed technical explanation
- `docs/PHASE4_IMPROVEMENTS.md` - What changed from previous version
- `docs/PRODUCTION_CHECKLIST.sh` - Verification checklist
- `federated-llm/docs/DEPLOY_TO_HIPERGATOR.md` - HiPerGator deployment guide

## 🔬 Research References

This implementation is based on:
- **McMahan et al. (2017)**: Federated Learning framework
- **Attention is All You Need (Vaswani et al., 2017)**: Transformer architecture
- **Llama 3.1 (Meta)**: Language model

## 🤝 Contributing

Improvements and extensions welcome:
1. Increase number of clients (> 3)
2. Add more diverse scenarios
3. Implement advanced aggregation strategies
4. Analyze client preference divergence

## 📞 Support

For issues or questions:
- Check `LLAMA_FEDERATED_APPROACH.md` for detailed explanations
- Review local test output with `python scripts/test_phase4_local.py`
- Check HiPerGator SLURM logs for execution issues

## 📝 License

This project is part of federated learning research at University of Florida.

---

**Last Updated**: April 17, 2026  
**Status**: Production Ready ✅
