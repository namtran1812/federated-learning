#!/bin/bash#!/bin/bash

#SBATCH --job-name=llama_federated#SBATCH --job-name=llama_federated

#SBATCH --output=llama_federated_%j.out#SBATCH --output=llama_federated_%j.out

#SBATCH --error=llama_federated_%j.err#SBATCH --error=llama_federated_%j.err

#SBATCH --nodes=1#SBATCH --nodes=1

#SBATCH --ntasks=1#SBATCH --ntasks=1

#SBATCH --cpus-per-task=8#SBATCH --cpus-per-task=8

#SBATCH --mem=64gb#SBATCH --mem=64gb

#SBATCH --time=02:00:00#SBATCH --time=02:00:00

#SBATCH --partition=hpg-default#SBATCH --partition=hpg-default



echo "Starting Llama-3.1-8B Federated Comparison on HiperGator"echo "Starting Llama-3.1-8B Federated Comparison on HiperGator"

echo "=========================================================="echo "=========================================================="

echo "Time: $(date)"echo "Time: $(date)"

echo ""echo ""



# Load Python module# Load Python module

module load python/3.11module load python/3.11



echo "Available Python:"echo "Available Python:"

which python3which python3

python3 --versionpython3 --version



echo ""echo ""

echo "Installing required packages..."echo "Installing required packages..."

pip install -q torch transformers bitsandbytes accelerate huggingface-hubpip install -q torch transformers bitsandbytes accelerate huggingface-hub



echo ""echo ""

echo "Setting HuggingFace token..."echo "Setting HuggingFace token..."

# Load from environment variable or use interactive login# Load from environment variable or use interactive login

# Authenticate with: huggingface-cli login# export HF_TOKEN="your_token_here"

# OR set export HF_TOKEN in your shell profile

echo ""

echo ""echo "Running Llama-3.1 Federated Comparison..."

echo "Running Llama-3.1 Federated Comparison..."cd /blue/jie.xu/nam.tran1/hipergator-project

cd /blue/jie.xu/nam.tran1/hipergator-project

# Run the script

# Run the scriptpython3 scripts/method_llama_3_1_combined.py

python3 scripts/method_llama_3_1_combined.py

echo ""

echo ""echo "Job completed at: $(date)"

echo "Job completed at: $(date)"
