#!/bin/bash
#SBATCH --job-name=interactive_fed
#SBATCH --output=interactive_fed_%j.out
#SBATCH --error=interactive_fed_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=00:30:00
#SBATCH --partition=hpg-default

echo "=================================================="
echo "INTERACTIVE FEDERATED LLM ON HIPERGATOR"
echo "=================================================="
echo "Time: $(date)"
echo ""

# Load GCC and Python
module load gcc/11.2.0
module load python/3.11

echo "Environment ready"
echo ""

cd /blue/jie.xu/nam.tran1/hipergator-project

# Install packages (no deps to avoid lib issues)
pip install -q --no-deps torch transformers

echo "Starting interactive session..."
echo ""

# Run interactive script with test input
python3 scripts/interactive_federated.py < test_questions.txt

echo ""
echo "Job completed at: $(date)"
