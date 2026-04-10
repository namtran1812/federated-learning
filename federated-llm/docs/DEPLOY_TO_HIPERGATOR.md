# Deploy to HiPerGator

## One Command
```bash
bash submit_to_hipergator.sh
```

## Manual Steps
```bash
# 1. Upload
scp -r . nam.tran1@hpg.rc.ufl.edu:/blue/jie.xu/nam.tran1/federated-llm/

# 2. SSH in
ssh nam.tran1@hpg.rc.ufl.edu
cd /blue/jie.xu/nam.tran1/federated-llm

# 3. Setup
module load gcc/11.2.0 python/3.11 cuda/12.2
python -m venv venv && source venv/bin/activate
pip install torch transformers accelerate huggingface-hub
huggingface-cli login

# 4. Run
sbatch slurm/phase_1.slurm
sbatch slurm/phase_2.slurm
sbatch slurm/phase_3.slurm
sbatch slurm/phase_4.slurm

# 5. Monitor
squeue -u nam.tran1
tail -f fed_phase_1_*.out

# 6. Download results
scp -r nam.tran1@hpg.rc.ufl.edu:/blue/jie.xu/nam.tran1/federated-llm/results ./
```

## Files
- `1_extract_token_scores.py` - What tokens does the model like?
- `2_compare_client_contexts.py` - Do different contexts matter?
- `3_aggregate_distributions.py` - Does averaging work?
- `4_generate_federated_text.py` - Can we generate full text?

## Runtime
Total: ~90 minutes on GPU
