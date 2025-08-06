#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=32G
#SBATCH --time=8:00:00

# Activate environment
source venv/bin/activate

# Base arguments (customize as needed)
MODEL="lgai/exaone-3-5-32b-instruct"
MAX_TOKENS=2048
NUM_RUNS=100
MIN_ELEMENTS=3
MAX_ELEMENTS=5
MIN_SWAPS=8
MAX_SWAPS=16
BRANCHING_FACTOR=2

# PrefixSumAgents
python chain_of_agents/run_permutations.py \
  --agent_type PrefixSumAgents \
  --model_type $MODEL \
  --max_tokens $MAX_TOKENS \
  --num_runs $NUM_RUNS \
  --min_elements $MIN_ELEMENTS \
  --max_elements $MAX_ELEMENTS \
  --min_swaps $MIN_SWAPS \
  --max_swaps $MAX_SWAPS \
  --branching_factor $BRANCHING_FACTOR