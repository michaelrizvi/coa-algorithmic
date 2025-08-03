# main.py
import argparse
import random
import logging
import wandb
from statistics import mean

from logger import setup_logger
from main import MajorityVotingAgents, ChainOfAgents, PrefixSumAgents  # Assume these are defined elsewhere
from utils import *
import tabulate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_type", choices=["MajorityVotingAgents", "ChainOfAgents", "PrefixSumAgents"], default="ChainOfAgents", help="Type of agent to use")
    parser.add_argument("--num_agents", type=int, default=5, help="Number of agents to use in MajVote setup")
    parser.add_argument("--model_type", type=str, default="lgai/exaone-3-5-32b-instruct", help="Model type to use for agents")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max tokens for each agent")
    parser.add_argument("--chunk_size", type=int, default=8, help="Chunk size for Chain of Agents")
    parser.add_argument("--num_runs", type=int, default=50, help="Number of runs to perform")
    parser.add_argument("--min_seq_length", type=int, default=6, help="Minimum sequence length for input")
    parser.add_argument("--max_seq_length", type=int, default=6, help="Maximum sequence length for input")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--index_hints", type=bool, default=False, help="Use index hints for the agents")
    parser.add_argument("--branching_factor", type=int, default=2, help="branching factor for prefix sum agents")
    args = parser.parse_args()

    # Create a nice table showing all arguments
    args_table = []
    for arg, value in vars(args).items():
        args_table.append([arg, value])
        
    print("\n" + "="*50)
    print("EXPERIMENT CONFIGURATION")
    print("="*50)
    print(tabulate.tabulate(args_table, headers=["Argument", "Value"], tablefmt="grid"))
    print("="*50 + "\n")
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    name_dict = {
        "MajorityVotingAgents": "maj-voting",
        "ChainOfAgents": "coa",
        "PrefixSumAgents": "prefix-sum"
    }
    run_name = f"{name_dict[args.agent_type]}_seq{args.min_seq_length}-{args.max_seq_length}_agents{args.num_agents}"
    wandb.init(project="coa-parity-eval", config=vars(args), name=run_name, reinit=True)
    logger = setup_logger()

    if args.agent_type == "MajorityVotingAgents":
        prompt = get_majority_vote_prompt(index_hints=args.index_hints)
        agent = MajorityVotingAgents(
            num_agents=args.num_agents,
            model=args.model_type,
            max_tokens=args.max_tokens,
            prompt=prompt,
        )
    elif args.agent_type == "ChainOfAgents":
        worker_prompt, manager_prompt = get_parity_prompt(index_hints=args.index_hints)
        agent = ChainOfAgents(
            worker_model=args.model_type,
            manager_model=args.model_type,
            chunk_size=args.chunk_size,
            worker_prompt=worker_prompt,
            manager_prompt=manager_prompt,
            max_tokens_worker=args.max_tokens,
            use_index_hints=args.index_hints,
        )
    elif args.agent_type == "PrefixSumAgents":
        worker_prompt, manager_prompt = get_prefix_sum_prompt(index_hints=args.index_hints)
        agent = PrefixSumAgents(
            worker_model=args.model_type,
            manager_model=args.model_type,
            max_tokens_worker=args.max_tokens,
            max_tokens_manager=args.max_tokens,
            worker_prompt=worker_prompt,
            manager_prompt=manager_prompt,
            branching_factor=args.branching_factor,
        )


    for seq_len in [2**exponent for exponent in range(args.min_seq_length, args.max_seq_length + 1)]:
        results = []
        for _ in range(args.num_runs):
            bitstring = generate_bitstring(seq_len, index_hints=args.index_hints)
            query = "What is the parity of the given binary string?"
            if args.index_hints:
                input_text = bitstring  
            else:
                input_text = " ".join(bitstring)  # Convert bitstring to space-separated

            if args.agent_type == "MajorityVotingAgents":
                pred = agent.process(input_text, query)
            elif args.agent_type == "ChainOfAgents":
                pred = agent.process(input_text, query, extraction_func=extract_answer)
            elif args.agent_type == "PrefixSumAgents":
                pred = agent.hierarchical_process(input_text, query, extraction_func=extract_answer)
            pred = pred.strip().lower()
            pred = "0" if pred == "even" else "1" if pred == "odd" else pred
            print(f"Bitstring: {bitstring}, Predicted Parity: {pred}")
            truth = compute_parity(bitstring)
            print(f"Truth Parity: {truth}")
            correct = int(pred == truth)
            results.append(correct)
        avg_accuracy = mean(results)
        max_accuracy = max(results)
        logger.info(f"SeqLen={seq_len}, AvgAccuracy={avg_accuracy:.3f}")
        logger.info(f"SeqLen={seq_len}, MaxAccuracy={max_accuracy:.3f}")
        wandb.log({"avg_accuracy": avg_accuracy, "sequence_length": seq_len})
        wandb.log({"max_accuracy": max_accuracy, "sequence_length": seq_len})

if __name__ == "__main__":
    main()
