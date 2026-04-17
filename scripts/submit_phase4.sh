#!/bin/bash

echo "=========================================================================="
echo "Uploading Phase 4 to HiPerGator and Submitting Job"
echo "=========================================================================="
echo ""

# Upload the updated code
echo "Uploading generate_federated_text.py to HiPerGator..."
scp federated-llm/src/generate_federated_text.py nam.tran1@hpg-login1.rc.ufl.edu:/blue/jie.xu/nam.tran1/federated-llm/src/

echo "Uploading SLURM script..."
scp federated-llm/slurm/4_generate_federated_text.slurm nam.tran1@hpg-login1.rc.ufl.edu:/blue/jie.xu/nam.tran1/federated-llm/slurm/

echo ""
echo "Submitting Phase 4 job to HiPerGator..."
ssh nam.tran1@hpg-login1.rc.ufl.edu "cd /blue/jie.xu/nam.tran1/federated-llm && sbatch slurm/4_generate_federated_text.slurm"

echo ""
echo "=========================================================================="
echo "Job submitted! Check progress with:"
echo "  ssh nam.tran1@hpg-login1.rc.ufl.edu 'squeue -u nam.tran1'"
echo "=========================================================================="
