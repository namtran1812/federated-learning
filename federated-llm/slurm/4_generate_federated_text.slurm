#!/bin/bash
#SBATCH --job-name=generate_federated_text
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

echo "==========================================================================="
echo "PHASE 4: Federated Text Generation with Llama-3.1-8B"
echo "==========================================================================="

cd /blue/jie.xu/nam.tran1/federated-llm

# Use bash to source conda properly
bash -l << 'EOFSCRIPT'
module load conda
conda activate llama_env 2>/dev/null || (echo "Creating conda environment..." && conda create -n llama_env python=3.10 pytorch::pytorch transformers accelerate -c pytorch -c conda-forge -y && conda activate llama_env)
echo "Starting Phase 4..."
python src/generate_federated_text.py
echo "Phase 4 complete!"
EOFSCRIPT
