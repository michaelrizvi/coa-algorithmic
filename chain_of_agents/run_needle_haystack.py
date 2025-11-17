"""
Needle-in-a-Haystack Benchmark for Multi-Agent Systems

This script evaluates different agent architectures on the needle-in-a-haystack task,
where a specific fact (needle) is hidden in a large corpus (haystack) and the model
must retrieve it accurately.

Agent types:
- MajorityVotingAgents: Self-consistency baseline (multiple independent agents vote)
- ChainOfAgents: Sequential processing with agent chaining

The script tests across:
- Various context lengths (1k, 2k, 4k, 8k, etc.)
- Different needle depths (0% to 100% through document)
- Multiple runs for statistical significance
"""

import argparse
import random
import logging
import wandb
from statistics import mean, stdev
from datetime import datetime
import math
import time

from logger import setup_logger
from main import MajorityVotingAgents, ChainOfAgents
from utils import (
    load_paul_graham_corpus,
    generate_needle_haystack_problem,
    extract_needle_answer,
    check_needle_answer_match,
    get_needle_haystack_prompts
)
import tabulate


def main():
    parser = argparse.ArgumentParser(description="Needle-in-a-Haystack benchmark for multi-agent systems")
    parser.add_argument("--agent_type", choices=["MajorityVotingAgents", "ChainOfAgents"],
                        default="ChainOfAgents", help="Type of agent to use")
    parser.add_argument("--num_agents", type=int, default=3,
                        help="Number of agents for MajorityVoting (default: 3)")
    parser.add_argument("--chunk_size", type=int, default=100,
                        help="Chunk size in words for ChainOfAgents (default: 1000)")
    parser.add_argument("--model_type", type=str, default="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                        help="Model type to use for agents")
    parser.add_argument("--max_tokens", type=int, default=1024,
                        help="Max tokens for each agent")
    parser.add_argument("--context_lengths", type=int, nargs='+', default=[1000, 2000, 4000, 8000],
                        help="Context lengths to test (in words)")
    parser.add_argument("--depths", type=float, nargs='+', default=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                        help="Needle depths to test (0.0=start, 1.0=end)")
    parser.add_argument("--num_runs", type=int, default=5,
                        help="Number of runs per configuration (default: 5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--corpus_path", type=str, default="data/paul_graham/essays.txt",
                        help="Path to Paul Graham essays corpus")
    args = parser.parse_args()

    # Create a nice table showing all arguments
    args_table = []
    for arg, value in vars(args).items():
        args_table.append([arg, value])

    print("\n" + "="*70)
    print("NEEDLE-IN-A-HAYSTACK EXPERIMENT CONFIGURATION")
    print("="*70)
    print(tabulate.tabulate(args_table, headers=["Argument", "Value"], tablefmt="grid"))
    print("="*70 + "\n")

    # Set random seed for reproducibility
    random.seed(args.seed)

    # Initialize W&B with unique, informative run name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ctx_range = f"{min(args.context_lengths)}-{max(args.context_lengths)}"

    if args.agent_type == "MajorityVotingAgents":
        run_name = f"needle_MV_a{args.num_agents}_ctx{ctx_range}_{timestamp}"
    elif args.agent_type == "ChainOfAgents":
        run_name = f"needle_CoA_c{args.chunk_size}_ctx{ctx_range}_{timestamp}"

    wandb.init(project="coa-needle-haystack-eval", config=vars(args), name=run_name, reinit=True)
    logger = setup_logger()

    # Load Paul Graham corpus
    logger.info(f"Loading corpus from {args.corpus_path}")
    corpus = load_paul_graham_corpus(args.corpus_path)
    logger.info(f"Corpus loaded: {len(corpus.split())} words")

    # Get prompts for needle-in-haystack task
    worker_prompt, manager_prompt = get_needle_haystack_prompts()

    # Initialize agent
    if args.agent_type == "MajorityVotingAgents":
        agent = MajorityVotingAgents(
            num_agents=args.num_agents,
            model=args.model_type,
            max_tokens=args.max_tokens,
            prompt=worker_prompt,
        )
        logger.info(f"Initialized MajorityVotingAgents with {args.num_agents} agents")
    elif args.agent_type == "ChainOfAgents":
        agent = ChainOfAgents(
            worker_model=args.model_type,
            manager_model=args.model_type,
            chunk_size=args.chunk_size,
            max_tokens_worker=args.max_tokens,
            max_tokens_manager=args.max_tokens,
            worker_prompt=worker_prompt,
            manager_prompt=manager_prompt,
        )
        logger.info(f"Initialized ChainOfAgents with chunk_size={args.chunk_size}")

    # Run experiments for different context lengths and depths
    total_experiments = len(args.context_lengths) * len(args.depths) * args.num_runs
    experiment_count = 0

    for context_length in args.context_lengths:
        for depth in args.depths:
            accuracy_results = []
            token_stats = []
            latency_stats = []

            for run_idx in range(args.num_runs):
                experiment_count += 1
                logger.info(f"\n{'='*70}")
                logger.info(f"Experiment {experiment_count}/{total_experiments}")
                logger.info(f"Context: {context_length} words, Depth: {depth:.1%}, Run: {run_idx+1}/{args.num_runs}")
                logger.info(f"{'='*70}")

                try:
                    # Generate needle-in-haystack problem
                    context_with_needle, query, ground_truth = generate_needle_haystack_problem(
                        corpus, context_length, depth
                    )

                    logger.info(f"Generated problem with {len(context_with_needle.split())} words")
                    logger.info(f"Query: {query}")
                    logger.info(f"Ground truth: {ground_truth}")

                    # Start timing
                    start_time = time.time()

                    # Get prediction from agent
                    result = agent.process(context_with_needle, query, extraction_func=extract_needle_answer)

                    # End timing
                    latency = time.time() - start_time
                    latency_stats.append(latency)

                    predicted_answer = result['content']
                    token_stats.append(result.get('token_usage', {}))

                    logger.info(f"Predicted answer: {predicted_answer}")
                    logger.info(f"Latency: {latency:.2f}s")

                    # Evaluate accuracy
                    correct = check_needle_answer_match(predicted_answer, ground_truth)
                    accuracy_results.append(int(correct))

                    logger.info(f"{'✓ CORRECT' if correct else '✗ INCORRECT'}")

                    # Log individual run to W&B
                    wandb.log({
                        "context_length": context_length,
                        "depth": depth,
                        "run_idx": run_idx,
                        "correct": int(correct),
                        "latency": latency,
                        "experiment_count": experiment_count,
                    })

                except Exception as e:
                    logger.error(f"Error in experiment {experiment_count}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    accuracy_results.append(0)
                    latency_stats.append(0)
                    token_stats.append({
                        'avg_completion_tokens': 0,
                        'max_completion_tokens': 0,
                        'avg_prompt_tokens': 0,
                        'max_prompt_tokens': 0
                    })

            # Calculate statistics for this configuration
            avg_accuracy = mean(accuracy_results)
            max_accuracy = max(accuracy_results)
            avg_latency = mean(latency_stats) if latency_stats else 0

            # Calculate standard error
            n_runs = len(accuracy_results)
            if n_runs > 1:
                std_accuracy = stdev(accuracy_results)
                se_accuracy = std_accuracy / math.sqrt(n_runs)
            else:
                std_accuracy = se_accuracy = 0.0

            logger.info(f"\n{'='*70}")
            logger.info(f"RESULTS: Context={context_length}, Depth={depth:.1%}")
            logger.info(f"Accuracy: {avg_accuracy:.3f} ± {se_accuracy:.3f} (max: {max_accuracy:.3f})")
            logger.info(f"Avg Latency: {avg_latency:.2f}s")
            logger.info(f"{'='*70}\n")

            # Calculate average token statistics
            if token_stats:
                avg_completion_tokens = mean([stats.get('avg_completion_tokens', 0) for stats in token_stats])
                max_completion_tokens = max([stats.get('max_completion_tokens', 0) for stats in token_stats])
                avg_prompt_tokens = mean([stats.get('avg_prompt_tokens', 0) for stats in token_stats])
                max_prompt_tokens = max([stats.get('max_prompt_tokens', 0) for stats in token_stats])
            else:
                # Fallback if token_stats is empty (shouldn't happen, but be safe)
                avg_completion_tokens = 0
                max_completion_tokens = 0
                avg_prompt_tokens = 0
                max_prompt_tokens = 0

            # ALWAYS log aggregated statistics to W&B (moved outside if block)
            # Log with commit=True to ensure immediate persistence (prevents sync issues)
            wandb.log({
                "agg_context_length": context_length,
                "agg_depth": depth,
                "agg_avg_accuracy": avg_accuracy,
                "agg_max_accuracy": max_accuracy,
                "agg_std_accuracy": std_accuracy,
                "agg_se_accuracy": se_accuracy,
                "agg_avg_latency": avg_latency,
                "agg_avg_completion_tokens": avg_completion_tokens,
                "agg_max_completion_tokens": max_completion_tokens,
                "agg_avg_prompt_tokens": avg_prompt_tokens,
                "agg_max_prompt_tokens": max_prompt_tokens,
            }, commit=True)

    logger.info("\n" + "="*70)
    logger.info("EXPERIMENT COMPLETE")
    logger.info(f"Total experiments run: {experiment_count}")
    logger.info(f"Results logged to W&B project: coa-needle-haystack-eval")
    logger.info("="*70)

    wandb.finish()


if __name__ == "__main__":
    main()
