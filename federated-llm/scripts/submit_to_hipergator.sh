#!/bin/bash
# Master script to set up and submit federated decoding research to HiPerGator

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║   Token-Level Federated Decoding with Llama-3.1-8B-Instruct          ║"
echo "║   Master Setup & Submission Script                                     ║"
echo "╚════════════════════════════════════════════════════════════════════════╝"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ============================================================================
# STEP 1: Verify Local Setup
# ============================================================================

echo -e "\n${BLUE}[STEP 1]${NC} Verifying local setup..."

if [ ! -f "test_setup.py" ]; then
    echo -e "${YELLOW}Warning: Run this script from the federated-llm directory${NC}"
    exit 1
fi

if ! command -v ssh &> /dev/null; then
    echo -e "${YELLOW}SSH not found. Please install OpenSSH.${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} Local setup verified"

# ============================================================================
# STEP 2: Run Quick Test
# ============================================================================

echo -e "\n${BLUE}[STEP 2]${NC} Running quick setup test (optional)..."
echo "Press Enter to skip test, or 'y' to run it:"
read -r run_test

if [ "$run_test" = "y" ] || [ "$run_test" = "Y" ]; then
    echo -e "${BLUE}Running test_setup.py...${NC}"
    if python3 test_setup.py; then
        echo -e "${GREEN}✓${NC} Setup test passed!"
    else
        echo -e "${YELLOW}Warning: Test failed. Continue anyway? (y/n)${NC}"
        read -r continue_anyway
        if [ "$continue_anyway" != "y" ]; then
            exit 1
        fi
    fi
else
    echo -e "${YELLOW}Skipping setup test${NC}"
fi

# ============================================================================
# STEP 3: Prepare HiPerGator Upload
# ============================================================================

echo -e "\n${BLUE}[STEP 3]${NC} Preparing files for HiPerGator upload..."

# Create tarball for easy transfer
TARBALL="federated-llm.tar.gz"
echo "Creating $TARBALL..."

tar -czf "$TARBALL" \
    README.md \
    EXECUTION_GUIDE.md \
    PROJECT_OVERVIEW.md \
    phase_*.py \
    test_setup.py \
    slurm/ \
    2>/dev/null

if [ -f "$TARBALL" ]; then
    SIZE=$(du -h "$TARBALL" | cut -f1)
    echo -e "${GREEN}✓${NC} Created $TARBALL ($SIZE)"
else
    echo -e "${YELLOW}Failed to create tarball${NC}"
    exit 1
fi

# ============================================================================
# STEP 4: Upload to HiPerGator
# ============================================================================

echo -e "\n${BLUE}[STEP 4]${NC} Uploading to HiPerGator..."

REMOTE_PATH="/blue/jie.xu/nam.tran1/federated-llm"
HPGATOR_HOST="nam.tran1@hpg.rc.ufl.edu"

echo "Uploading to $HPGATOR_HOST:$REMOTE_PATH"
echo "(You will be prompted for Duo two-factor authentication)"

if scp "$TARBALL" "$HPGATOR_HOST:~/" && \
   ssh "$HPGATOR_HOST" "mkdir -p $REMOTE_PATH && cd ~ && tar -xzf $TARBALL -C $REMOTE_PATH && rm $TARBALL"; then
    echo -e "${GREEN}✓${NC} Upload successful!"
else
    echo -e "${YELLOW}Upload failed${NC}"
    exit 1
fi

# ============================================================================
# STEP 5: Set Up HiPerGator Environment
# ============================================================================

echo -e "\n${BLUE}[STEP 5]${NC} Setting up HiPerGator environment..."

setup_cmd='
cd /blue/jie.xu/nam.tran1/federated-llm
module load gcc/11.2.0 python/3.11 cuda/12.2
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch transformers accelerate numpy
mkdir -p results
echo "Environment setup complete!"
'

if ssh "$HPGATOR_HOST" "$setup_cmd"; then
    echo -e "${GREEN}✓${NC} HiPerGator environment ready!"
else
    echo -e "${YELLOW}Environment setup failed${NC}"
    exit 1
fi

# ============================================================================
# STEP 6: Submit Phase 1 Job
# ============================================================================

echo -e "\n${BLUE}[STEP 6]${NC} Submitting Phase 1 job to HiPerGator..."
echo "This will:"
echo "  1. Extract token distributions from 3 test prompts"
echo "  2. Show top-10 tokens with probabilities at each decoding step"
echo "  3. Save results to results/phase_1_scores.json"
echo ""
echo "Submit now? (y/n)"
read -r submit_now

if [ "$submit_now" = "y" ] || [ "$submit_now" = "Y" ]; then
    
    submit_cmd='
    cd /blue/jie.xu/nam.tran1/federated-llm
    source venv/bin/activate
    JOB_ID=$(sbatch slurm/phase_1.slurm | awk "{print \$NF}")
    echo "Job submitted with ID: $JOB_ID"
    echo "Check status with: squeue -j $JOB_ID"
    '
    
    if ssh "$HPGATOR_HOST" "$submit_cmd"; then
        echo -e "${GREEN}✓${NC} Phase 1 job submitted!"
        echo ""
        echo "Next steps:"
        echo "  1. Monitor job: ssh nam.tran1@hpg.rc.ufl.edu 'squeue -u nam.tran1'"
        echo "  2. View logs: ssh nam.tran1@hpg.rc.ufl.edu 'tail -f /blue/jie.xu/nam.tran1/federated-llm/fed_phase_1_*.out'"
        echo "  3. Check results: ssh nam.tran1@hpg.rc.ufl.edu 'cat /blue/jie.xu/nam.tran1/federated-llm/results/phase_1_scores.json'"
        echo ""
        echo "For more details, see EXECUTION_GUIDE.md"
    else
        echo -e "${YELLOW}Job submission failed${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}Job submission skipped${NC}"
    echo "To submit later, run on HiPerGator:"
    echo "  sbatch /blue/jie.xu/nam.tran1/federated-llm/slurm/phase_1.slurm"
fi

# ============================================================================
# FINAL SUMMARY
# ============================================================================

echo ""
echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║                        Setup Complete! ✓                              ║"
echo "╚════════════════════════════════════════════════════════════════════════╝"

echo ""
echo "📊 Project Structure:"
echo "  Remote path: /blue/jie.xu/nam.tran1/federated-llm"
echo ""
echo "📁 Main Files:"
echo "  • README.md - Research background"
echo "  • PROJECT_OVERVIEW.md - This project explained"
echo "  • EXECUTION_GUIDE.md - Detailed HiPerGator instructions"
echo "  • phase_1_extract_scores.py - Extract token distributions"
echo "  • phase_2_multi_client_contexts.py - Multi-client simulation"
echo "  • phase_3_aggregate_distributions.py - Distribution aggregation"
echo "  • phase_4_autoregressive_decoding.py - Full generation"
echo ""
echo "⚡ Quick Commands:"
echo "  Check job status:  ssh nam.tran1@hpg.rc.ufl.edu 'squeue -u nam.tran1'"
echo "  View output:       ssh nam.tran1@hpg.rc.ufl.edu 'tail -f /blue/jie.xu/nam.tran1/federated-llm/fed_phase_*.out'"
echo "  Run next phase:    ssh nam.tran1@hpg.rc.ufl.edu 'sbatch /blue/jie.xu/nam.tran1/federated-llm/slurm/phase_2.slurm'"
echo ""
echo "📚 For more information:"
echo "  • See EXECUTION_GUIDE.md for step-by-step instructions"
echo "  • See README.md for research details"
echo "  • See PROJECT_OVERVIEW.md for quick overview"
echo ""
echo "🚀 You're ready to run token-level federated decoding research!"
echo ""
