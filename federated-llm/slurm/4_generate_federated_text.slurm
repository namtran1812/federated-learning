#!/bin/bash
#SBATCH --job-name=generate_federated_text
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=03:00:00
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

echo "Starting Phase 4: Autoregressive Federated Decoding"
python3 src/generate_federated_text.py

echo "Phase 4 complete!"
