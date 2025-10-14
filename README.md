# Chain-of-Agents: Multi-Agent Reasoning Systems

This repository contains the codebase for the paper:

**"BENEFITS AND LIMITATIONS OF COMMUNICATION IN MULTI-AGENT REASONING"**

## Abstract

Chain-of-thought prompting has popularized step-by-step reasoning in large language models, yet model performance still degrades as problem complexity and context length grow. By decomposing difficult tasks with long contexts into shorter, manageable ones, recent multi-agent paradigms offer a promising near-term solution to this problem. However, the fundamental capacities of such systems are poorly understood. In this work, we propose a theoretical framework to analyze the expressivity of multi-agent systems. We apply our framework to three algorithmic families: state tracking, recall, and $k$-hop reasoning. We derive bounds on (i) the number of agents required to solve the task exactly, (ii) the quantity and structure of inter-agent communication, and (iii) the achievable speedups as problem size and context scale. Our results identify regimes where communication is provably beneficial, delineate tradeoffs between agent count and bandwidth, and expose intrinsic limitations when either resource is constrained. We complement our theoretical analysis with a set of experiments on pretrained LLMs using controlled synthetic benchmarks. Empirical outcomes confirm the tradeoffs between key quantities predicted by our theory. Collectively, our analysis offers principled guidance for designing scalable multi-agent reasoning systems.

## Overview

This codebase implements and evaluates multiple multi-agent reasoning architectures on controlled synthetic benchmarks. The experiments validate theoretical predictions about the tradeoffs between agent count, communication bandwidth, and task performance across three algorithmic families:

- **State Tracking**: Permutation tracking and parity computation tasks
- **Recall**: K-hop reasoning over knowledge graphs
- **Pareto Analysis**: Tradeoff curves between performance and resource usage

## Installation

### Prerequisites
- Python 3.9+
- Together AI API key

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Chain-of-Agents.git
cd Chain-of-Agents
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your API key:
Create a `.env` file in the root directory with your Together AI API key:
```
TOGETHER_API_KEY=your_api_key_here
```

5. Configure Weights & Biases (W&B):
```bash
wandb login
```
All experiments log results to W&B for tracking and visualization.

## Repository Structure

```
Chain-of-Agents/
├── chain_of_agents/
│   ├── main.py              # Core agent implementations
│   ├── agents.py            # Agent base classes
│   ├── utils.py             # Task generation and evaluation utilities
│   ├── logger.py            # Logging configuration
│   ├── run_khop.py          # K-hop reasoning experiments
│   ├── run_khop_pareto.py   # K-hop Pareto analysis
│   ├── run_parity.py        # Parity computation experiments
│   ├── run_pareto.py        # Parity Pareto analysis
│   ├── run_permutations.py  # Permutation tracking experiments
│   └── run_pareto_permutations.py  # Permutation Pareto analysis
├── data/                    # Experiment data and results
├── figures/                 # Generated plots and visualizations
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Agent Architectures

The codebase implements several multi-agent architectures:

- **MajorityVotingAgents**: Multiple independent agents process the full input; final answer determined by majority vote
- **ChainOfAgents**: Sequential processing where each agent handles a chunk of the input and passes results to the next
- **PrefixSumAgents**: Hierarchical tree-based aggregation with configurable branching factor
- **IterativeQueryAgents**: Manager-worker architecture where workers process subsets of data and a manager aggregates results

## Experiments

All experiment scripts support various command-line arguments for configuration. Run any script with `--help` to see all available options.

### 1. K-Hop Reasoning (`run_khop.py`)

Evaluates multi-agent systems on k-hop reasoning tasks over synthetic knowledge graphs. Agents must traverse k relationships to answer queries.

**Key Arguments:**
- `--agent_type`: Choose from `MajorityVotingAgents` or `IterativeQueryAgents`
- `--model_type`: LLM model to use (default: `meta-llama/Llama-3.3-70B-Instruct-Turbo`)
- `--num_agents`: Number of agents for majority voting (default: 3)
- `--facts_per_worker`: Facts allocated to each worker for iterative queries (default: 20)
- `--min_hops` / `--max_hops`: Range of hop counts to evaluate (default: 20)
- `--num_facts`: Total number of facts in the knowledge graph (default: 100)
- `--num_runs`: Number of repetitions per configuration (default: 5)
- `--step`: Step size for incrementing hops (default: 2)

**Example Usage:**
```bash
# Iterative query agents on k-hop reasoning
python chain_of_agents/run_khop.py \
  --agent_type IterativeQueryAgents \
  --facts_per_worker 20 \
  --min_hops 10 \
  --max_hops 30 \
  --num_facts 200 \
  --num_runs 10

# Majority voting agents
python chain_of_agents/run_khop.py \
  --agent_type MajorityVotingAgents \
  --num_agents 5 \
  --min_hops 10 \
  --max_hops 30 \
  --num_facts 200
```

**Output:** Logs accuracy, token usage, and performance metrics to W&B project `coa-khop-eval`.

---

### 2. K-Hop Pareto Analysis (`run_khop_pareto.py`)

Analyzes the Pareto frontier for k-hop reasoning by varying the `facts_per_worker` parameter to explore accuracy-efficiency tradeoffs.

**Key Arguments:**
- `--model_type`: LLM model to use
- `--nb_hops`: Fixed number of hops for the experiment (default: 5)
- `--nb_facts`: Total number of facts (default: 500)
- `--min_facts_per_worker` / `--max_facts_per_worker`: Range of facts per worker (default: 20-100)
- `--step`: Step size for facts_per_worker (default: 20)
- `--num_runs`: Number of repetitions (default: 5)

**Example Usage:**
```bash
python chain_of_agents/run_khop_pareto.py \
  --nb_hops 10 \
  --nb_facts 500 \
  --min_facts_per_worker 25 \
  --max_facts_per_worker 125 \
  --step 25 \
  --num_runs 10
```

**Output:** Logs Pareto frontier data to W&B project `coa-khop-pareto-eval`.

---

### 3. Parity Computation (`run_parity.py`)

Evaluates agents on computing prefix sums or parity over binary sequences.

**Key Arguments:**
- `--agent_type`: Choose from `MajorityVotingAgents`, `ChainOfAgents`, or `PrefixSumAgents`
- `--model_type`: LLM model to use (default: `lgai/exaone-3-5-32b-instruct`)
- `--num_agents`: Number of agents for majority voting (default: 4)
- `--chunk_size`: Chunk size for chain of agents (default: 2)
- `--branching_factor`: Branching factor for prefix sum agents (default: 2)
- `--min_seq_length` / `--max_seq_length`: Range of sequence lengths (default: 3)
- `--num_runs`: Number of repetitions (default: 5)

**Example Usage:**
```bash
# Prefix sum agents
python chain_of_agents/run_parity.py \
  --agent_type PrefixSumAgents \
  --branching_factor 4 \
  --min_seq_length 8 \
  --max_seq_length 32 \
  --num_runs 10

# Chain of agents
python chain_of_agents/run_parity.py \
  --agent_type ChainOfAgents \
  --chunk_size 4 \
  --min_seq_length 8 \
  --max_seq_length 32
```

**Output:** Logs accuracy and token metrics to W&B project `coa-parity-eval`.

---

### 4. Parity Pareto Analysis (`run_pareto.py`)

Explores the Pareto frontier for prefix sum computation by varying the branching factor.

**Key Arguments:**
- `--model_type`: LLM model to use (default: `openai/gpt-oss-20b`)
- `--seq_length`: Fixed sequence length (default: 32)
- `--min_branching_factor` / `--max_branching_factor`: Range of branching factors (default: 2)
- `--step`: Step size for branching factor (default: 2)
- `--num_runs`: Number of repetitions (default: 2)

**Example Usage:**
```bash
python chain_of_agents/run_pareto.py \
  --seq_length 64 \
  --min_branching_factor 2 \
  --max_branching_factor 8 \
  --step 2 \
  --num_runs 5
```

**Output:** Logs Pareto data to W&B project `coa-pareto-eval`.

---

### 5. Permutation Tracking (`run_permutations.py`)

Evaluates agents on tracking ball positions through sequences of swaps.

**Key Arguments:**
- `--agent_type`: Choose from `MajorityVotingAgents`, `ChainOfAgents`, or `PrefixSumAgents`
- `--model_type`: LLM model to use (default: `meta-llama/Llama-3.3-70B-Instruct-Turbo`)
- `--num_elements`: Number of elements to permute (default: 5)
- `--num_agents`: Number of agents for majority voting (default: 3)
- `--chunk_size`: Chunk size for chain of agents (default: 2)
- `--branching_factor`: Branching factor for prefix sum agents (default: 2)
- `--min_swaps` / `--max_swaps`: Range of swap counts (default: 4-12)
- `--a5_only`: Constrain to even permutations only (default: True)
- `--step`: Step size for swaps (default: 2)
- `--num_runs`: Number of repetitions (default: 5)

**Example Usage:**
```bash
# Prefix sum agents on permutations
python chain_of_agents/run_permutations.py \
  --agent_type PrefixSumAgents \
  --branching_factor 2 \
  --num_elements 5 \
  --min_swaps 6 \
  --max_swaps 20 \
  --a5_only True \
  --num_runs 10

# Chain of agents
python chain_of_agents/run_permutations.py \
  --agent_type ChainOfAgents \
  --chunk_size 3 \
  --num_elements 5 \
  --min_swaps 6 \
  --max_swaps 20
```

**Output:** Logs exact match accuracy, element-wise accuracy, and token usage to W&B project `coa-permutation-eval`.

---

### 6. Permutation Pareto Analysis (`run_pareto_permutations.py`)

Analyzes the Pareto frontier for permutation tracking by varying branching factor.

**Key Arguments:**
- `--model_type`: LLM model to use (default: `lgai/exaone-3-5-32b-instruct`)
- `--num_elements`: Number of elements (default: 5)
- `--num_swaps`: Fixed number of swaps (default: 10)
- `--min_branching_factor` / `--max_branching_factor`: Range of branching factors (default: 2-4)
- `--step`: Step size for branching factor (default: 2)
- `--num_runs`: Number of repetitions (default: 10)

**Example Usage:**
```bash
python chain_of_agents/run_pareto_permutations.py \
  --num_elements 5 \
  --num_swaps 15 \
  --min_branching_factor 2 \
  --max_branching_factor 8 \
  --step 2 \
  --num_runs 10
```

**Output:** Logs Pareto frontier data to W&B project `coa-pareto-permutation-eval`.

---

## Results and Analysis

Experimental results are logged to Weights & Biases and can be visualized using the provided Jupyter notebooks:
- `khop-notebook.ipynb`: K-hop reasoning analysis
- `parity-notebook.ipynb`: Parity computation analysis
- `permutations-notebook.ipynb`: Permutation tracking analysis
- `pareto-notebook.ipynb`: Pareto frontier visualizations

Generated figures are saved in the `figures/` directory.

## License

See [LICENSE](LICENSE) for details.

## Contact

For questions or issues, please open an issue on the GitHub repository.
