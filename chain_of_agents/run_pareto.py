import argparse
import random
import logging
import wandb
from statistics import mean, stdev
import math

from logger import setup_logger
from main import PrefixSumAgents
from utils import *
import tabulate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="openai/gpt-oss-20b", help="Model type to use for agents")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max tokens for each agent")
    parser.add_argument("--num_runs", type=int, default=2, help="Number of runs to perform")
    parser.add_argument("--seq_length", type=int, default=32, help="Fixed sequence length for input")
    parser.add_argument("--min_branching_factor", type=int, default=2, help="Minimum branching factor")
    parser.add_argument("--max_branching_factor", type=int, default=2, help="Maximum branching factor")
    parser.add_argument("--step", type=int, default=2, help="Step size for branching factor")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--index_hints", type=bool, default=False, help="Use index hints for the agents")
    args = parser.parse_args()

    # Create a nice table showing all arguments
    args_table = []
    for arg, value in vars(args).items():
        args_table.append([arg, value])
        
    print("\n" + "="*50)
    print("PARETO EXPERIMENT CONFIGURATION")
    print("="*50)
    print(tabulate.tabulate(args_table, headers=["Argument", "Value"], tablefmt="grid"))
    print("="*50 + "\n")
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    run_name = f"pareto_prefixsum_seq{args.seq_length}_b{args.min_branching_factor}-{args.max_branching_factor}"
    wandb.init(project="coa-pareto-eval", config=vars(args), name=run_name, reinit=True)
    logger = setup_logger()

    # Loop over branching factors
    for branching_factor in range(args.min_branching_factor, args.max_branching_factor + 1, args.step):
        # Initialize PrefixSumAgents for this branching factor
        worker_prompt, manager_prompt = get_prefix_sum_prompt(index_hints=args.index_hints)
        agent = PrefixSumAgents(
            worker_model=args.model_type,
            manager_model=args.model_type,
            max_tokens_worker=args.max_tokens,
            max_tokens_manager=args.max_tokens,
            worker_prompt=worker_prompt,
            manager_prompt=manager_prompt,
            branching_factor=branching_factor,
        )

        results = []
        token_stats = []
        
        for run_idx in range(args.num_runs):
            bitstring = generate_bitstring(args.seq_length, index_hints=args.index_hints)
            query = "What is the parity of the given binary string?"
            if args.index_hints:
                input_text = bitstring  
            else:
                input_text = " ".join(bitstring)  # Convert bitstring to space-separated

            result = agent.hierarchical_process(input_text, query, extraction_func=extract_answer)
            
            # Handle case where result['content'] is None
            if result['content'] is None:
                pred = "unknown"
                logger.warning(f"BranchingFactor={branching_factor}, Run={run_idx+1}, Result content is None, setting pred to 'unknown'")
            else:
                pred = result['content'].strip().lower()
                pred = "0" if pred == "even" else "1" if pred == "odd" else pred
            token_stats.append(result['token_usage'])
            
            logger.info(f"BranchingFactor={branching_factor}, Run={run_idx+1}, Bitstring: {bitstring[:20]}..., Predicted Parity: {pred}")
            truth = compute_parity(bitstring)
            logger.info(f"BranchingFactor={branching_factor}, Run={run_idx+1}, Truth Parity: {truth}")
            correct = int(pred == truth)
            results.append(correct)
            logger.info(f"BranchingFactor={branching_factor}, Run={run_idx+1}, Correct: {correct}")
        
        # Calculate statistics for this branching factor
        avg_accuracy = mean(results)
        max_accuracy = max(results)
        
        # Calculate standard error (SE = std_dev / sqrt(n))
        n_runs = len(results)
        if n_runs > 1:
            std_accuracy = stdev(results)
            se_accuracy = std_accuracy / math.sqrt(n_runs)
        else:
            std_accuracy = se_accuracy = 0.0
        
        logger.info(f"BranchingFactor={branching_factor}, AvgAccuracy={avg_accuracy:.3f}±{se_accuracy:.3f}")
        logger.info(f"BranchingFactor={branching_factor}, MaxAccuracy={max_accuracy:.3f}")
        
        # Calculate average token statistics for this branching factor
        if token_stats:
            avg_completion_tokens = mean([stats['avg_completion_tokens'] for stats in token_stats])
            max_completion_tokens = max([stats['max_completion_tokens'] for stats in token_stats])
            avg_prompt_tokens = mean([stats['avg_prompt_tokens'] for stats in token_stats])
            max_prompt_tokens = max([stats['max_prompt_tokens'] for stats in token_stats])
            
            # Calculate per-agent token statistics
            mean_completion_tokens_per_agent = mean([stats['mean_completion_tokens_per_agent'] for stats in token_stats])
            max_completion_tokens_per_agent = max([stats['max_completion_tokens_per_agent'] for stats in token_stats])
            mean_prompt_tokens_per_agent = mean([stats['mean_prompt_tokens_per_agent'] for stats in token_stats])
            max_prompt_tokens_per_agent = max([stats['max_prompt_tokens_per_agent'] for stats in token_stats])
            
            logger.info(f"BranchingFactor={branching_factor}, AvgCompletionTokens={avg_completion_tokens:.2f}")
            logger.info(f"BranchingFactor={branching_factor}, MaxCompletionTokens={max_completion_tokens:.2f}")
            logger.info(f"BranchingFactor={branching_factor}, AvgPromptTokens={avg_prompt_tokens:.2f}")
            logger.info(f"BranchingFactor={branching_factor}, MaxPromptTokens={max_prompt_tokens:.2f}")
            logger.info(f"BranchingFactor={branching_factor}, MeanCompletionTokensPerAgent={mean_completion_tokens_per_agent:.2f}")
            logger.info(f"BranchingFactor={branching_factor}, MaxCompletionTokensPerAgent={max_completion_tokens_per_agent:.2f}")
            logger.info(f"BranchingFactor={branching_factor}, MeanPromptTokensPerAgent={mean_prompt_tokens_per_agent:.2f}")
            logger.info(f"BranchingFactor={branching_factor}, MaxPromptTokensPerAgent={max_prompt_tokens_per_agent:.2f}")
            
            wandb.log({
                "avg_accuracy": avg_accuracy, 
                "max_accuracy": max_accuracy,
                "std_accuracy": std_accuracy,
                "se_accuracy": se_accuracy,
                "avg_completion_tokens": avg_completion_tokens,
                "max_completion_tokens": max_completion_tokens,
                "avg_prompt_tokens": avg_prompt_tokens,
                "max_prompt_tokens": max_prompt_tokens,
                "mean_completion_tokens_per_agent": mean_completion_tokens_per_agent,
                "max_completion_tokens_per_agent": max_completion_tokens_per_agent,
                "mean_prompt_tokens_per_agent": mean_prompt_tokens_per_agent,
                "max_prompt_tokens_per_agent": max_prompt_tokens_per_agent,
                "branching_factor": branching_factor,
                "sequence_length": args.seq_length
            })
        else:
            wandb.log({
                "avg_accuracy": avg_accuracy, 
                "max_accuracy": max_accuracy,
                "std_accuracy": std_accuracy,
                "se_accuracy": se_accuracy,
                "branching_factor": branching_factor,
                "sequence_length": args.seq_length
            })


if __name__ == "__main__":
    main()