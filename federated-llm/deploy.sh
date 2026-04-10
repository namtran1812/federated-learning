#!/bin/bash
# Deploy to HiPerGator - One Command

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║   DEPLOYING FEDERATED LLM TO HIPERGATOR                       ║"
echo "╚════════════════════════════════════════════════════════════════╝"

REMOTE_USER="nam.tran1"
REMOTE_HOST="hpg.rc.ufl.edu"
REMOTE_PATH="/blue/jie.xu/nam.tran1/federated-llm"

# Step 1: Upload to HiPerGator
echo ""
echo "[1/4] Uploading project to HiPerGator..."
scp -r /Users/namtran/LLM/federated-llm ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH} 2>/dev/null || {
    echo "Uploading..."
    rsync -avz /Users/namtran/LLM/federated-llm/ ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/
}
echo "✓ Uploaded"

# Step 2: Setup environment on HiPerGator
echo ""
echo "[2/4] Setting up environment on HiPerGator..."
ssh ${REMOTE_USER}@${REMOTE_HOST} << 'REMOTE_SETUP'
cd /blue/jie.xu/nam.tran1/federated-llm

# Load modules
module load gcc/11.2.0 python/3.11 cuda/12.2

# Create venv if needed
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Activate and install
source venv/bin/activate
pip install -q torch transformers accelerate huggingface-hub 2>/dev/null

echo "✓ Environment ready"
REMOTE_SETUP

# Step 3: Submit all jobs
echo ""
echo "[3/4] Submitting jobs to SLURM..."
ssh ${REMOTE_USER}@${REMOTE_HOST} << 'REMOTE_JOBS'
cd /blue/jie.xu/nam.tran1/federated-llm

JOB1=$(sbatch slurm/1_extract_token_scores.slurm 2>&1 | grep -oE '[0-9]+' | head -1)
echo "  Job 1 (Extract): $JOB1"

JOB2=$(sbatch --dependency=afterok:$JOB1 slurm/2_compare_client_contexts.slurm 2>&1 | grep -oE '[0-9]+' | head -1)
echo "  Job 2 (Compare): $JOB2"

JOB3=$(sbatch --dependency=afterok:$JOB2 slurm/3_aggregate_distributions.slurm 2>&1 | grep -oE '[0-9]+' | head -1)
echo "  Job 3 (Aggregate): $JOB3"

JOB4=$(sbatch --dependency=afterok:$JOB3 slurm/4_generate_federated_text.slurm 2>&1 | grep -oE '[0-9]+' | head -1)
echo "  Job 4 (Generate): $JOB4"

echo ""
echo "Jobs submitted with dependencies (sequential execution)"
REMOTE_JOBS

# Step 4: Show monitoring instructions
echo ""
echo "[4/4] Monitoring..."
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║   DEPLOYMENT COMPLETE                                         ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Monitor progress:"
echo "  ssh ${REMOTE_USER}@${REMOTE_HOST}"
echo "  cd ${REMOTE_PATH}"
echo "  squeue -u ${REMOTE_USER}"
echo "  tail -f *.out"
echo ""
echo "Download results when done:"
echo "  scp -r ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/results ./"
echo ""
echo "Expected runtime: ~90 minutes total"
echo "  Step 1: 10 min"
echo "  Step 2: 20 min"
echo "  Step 3: 15 min"
echo "  Step 4: 30 min"
echo ""
