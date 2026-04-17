#!/bin/bash

set -e

echo "=========================================================================="
echo "Phase 4 Federated Text Generation - HiPerGator Submission"
echo "=========================================================================="
echo ""

# Step 1: Upload the Python script
echo "[1/4] Uploading generate_federated_text.py..."
scp /Users/namtran/LLM/federated-llm/src/generate_federated_text.py \
    nam.tran1@hpg.rc.ufl.edu:/blue/jie.xu/nam.tran1/federated-llm/src/ 2>&1

echo "[2/4] Submitting SLURM job..."
JOBID=$(ssh nam.tran1@hpg.rc.ufl.edu << 'SSH_COMMAND'
cd /blue/jie.xu/nam.tran1/federated-llm
sbatch slurm/4_generate_federated_text.slurm 2>&1 | grep -oP 'Submitted batch job \K[0-9]+'
SSH_COMMAND
)

if [ -z "$JOBID" ]; then
    echo "❌ Failed to submit job"
    exit 1
fi

echo "✅ Job submitted with ID: $JOBID"
echo ""

# Step 3: Monitor job
echo "[3/4] Job Status:"
ssh nam.tran1@hpg.rc.ufl.edu "squeue -j $JOBID --format='%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R'"
echo ""

echo "[4/4] Results will be saved to:"
echo "  /blue/jie.xu/nam.tran1/federated-llm/results/phase_4_generations.json"
echo ""

echo "=========================================================================="
echo "✅ Phase 4 is now running on HiPerGator!"
echo "=========================================================================="
echo ""
echo "Monitor the job anytime with:"
echo "  ssh nam.tran1@hpg.rc.ufl.edu 'squeue -j $JOBID'"
echo ""
echo "After job completes (~20-30 mins), download results:"
echo "  scp nam.tran1@hpg.rc.ufl.edu:/blue/jie.xu/nam.tran1/federated-llm/results/phase_4_generations.json \\"
echo "      /Users/namtran/LLM/federated-llm/results/"
echo ""
