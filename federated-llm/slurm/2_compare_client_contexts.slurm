#!/bin/bash
#SBATCH --job-name=compare_client_contexts
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=02:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -e  # Exit on any error

cd /blue/jie.xu/nam.tran1/federated-llm

# Use system python and install to user dir
export PYTHONUSERBASE=$HOME/.local
export PATH=$PYTHONUSERBASE/bin:$PATH

echo "Installing dependencies..."
python3 -m pip install --upgrade pip --quiet 2>/dev/null || true
python3 -m pip install torch transformers accelerate --quiet 2>/dev/null || true

echo "Starting Phase 2: Multi-Client Contexts"
python3 src/compare_client_contexts.py

echo "Phase 2 complete!"
