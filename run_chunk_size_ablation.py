"""
Chunk Size Ablation Study for Chain-of-Agents

This script evaluates ChainOfAgents performance across different chunk sizes
on the needle-in-a-haystack task with fixed context length and needle depth.

The script tests:
- Various chunk sizes (specified via --chunk_sizes argument)
- Fixed context length and needle depth
- Multiple runs for statistical significance

Logs to W&B:
- Accuracy per chunk size
- Token usage (computation depth) per chunk size
"""

import argparse
import random
import logging
import wandb
from statistics import mean, stdev
from datetime import datetime
import math
import sys
import os

# Add chain_of_agents to path if running from root directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'chain_of_agents'))

from logger import setup_logger
from main import ChainOfAgents
from utils import (
    load_paul_graham_corpus,
    generate_needle_haystack_problem,
    extract_needle_answer,
    check_needle_answer_match,
    get_needle_haystack_prompts
)
import tabulate


def main():
    parser = argparse.ArgumentParser(description="Chunk size ablation for Chain-of-Agents")
    parser.add_argument("--chunk_sizes", type=int, nargs='+', required=True,
                        help="Chunk sizes to test (e.g., --chunk_sizes 50 100 200 400)")
    parser.add_argument("--context_length", type=int, default=4000,
                        help="Fixed context length in words (default: 4000)")
    parser.add_argument("--depth", type=float, default=0.5,
                        help="Fixed needle depth (0.0=start, 1.0=end, default: 0.5)")
    parser.add_argument("--model_type", type=str, default="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                        help="Model type to use for agents")
    parser.add_argument("--max_tokens", type=int, default=1024,
                        help="Max tokens for each agent")
    parser.add_argument("--num_runs", type=int, default=5,
                        help="Number of runs per chunk size (default: 5)")
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
    print("CHUNK SIZE ABLATION EXPERIMENT CONFIGURATION")
    print("="*70)
    print(tabulate.tabulate(args_table, headers=["Argument", "Value"], tablefmt="grid"))
    print("="*70 + "\n")

    # Set random seed for reproducibility
    random.seed(args.seed)

    # Initialize W&B with unique, informative run name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    chunk_range = f"{min(args.chunk_sizes)}-{max(args.chunk_sizes)}"
    run_name = f"chunk_ablation_c{chunk_range}_ctx{args.context_length}_d{args.depth}_{timestamp}"

    # Initialize W&B with settings optimized for SLURM/cluster environments
    wandb.init(
        project="coa-chunk-ablation",
        config=vars(args),
        name=run_name,
        reinit=True,
        settings=wandb.Settings(
            start_method="thread",  # Better for cluster environments
            _disable_stats=True,     # Reduce overhead on compute clusters
        )
    )

    # Define metrics with proper x-axis relationships
    # This helps W&B organize data correctly and improves sync reliability
    wandb.define_metric("experiment_count")  # Our global step counter
    wandb.define_metric("individual/*", step_metric="experiment_count")  # Individual run metrics
    wandb.define_metric("agg/*", step_metric="experiment_count")  # Aggregated metrics

    logger = setup_logger(enable_wandb=False)  # Disable wandb logging to prevent excessive API calls

    # Load Paul Graham corpus
    logger.info(f"Loading corpus from {args.corpus_path}")
    corpus = load_paul_graham_corpus(args.corpus_path)
    logger.info(f"Corpus loaded: {len(corpus.split())} words")

    # Get prompts for needle-in-haystack task
    worker_prompt, manager_prompt = get_needle_haystack_prompts()

    # Storage for final results table
    results_table = []

    # Run experiments for different chunk sizes
    total_experiments = len(args.chunk_sizes) * args.num_runs
    experiment_count = 0

    try:
        for chunk_size in args.chunk_sizes:
            # Initialize agent with current chunk size
            agent = ChainOfAgents(
                worker_model=args.model_type,
                manager_model=args.model_type,
                chunk_size=chunk_size,
                max_tokens_worker=args.max_tokens,
                max_tokens_manager=args.max_tokens,
                worker_prompt=worker_prompt,
                manager_prompt=manager_prompt,
            )
            logger.info(f"Initialized ChainOfAgents with chunk_size={chunk_size}")

            accuracy_results = []
            token_stats = []
            latency_stats = []

            for run_idx in range(args.num_runs):
                experiment_count += 1
                logger.info(f"\n{'='*70}")
                logger.info(f"Experiment {experiment_count}/{total_experiments}")
                logger.info(f"Chunk Size: {chunk_size}, Run: {run_idx+1}/{args.num_runs}")
                logger.info(f"{'='*70}")

                try:
                    # Generate needle-in-haystack problem
                    context_with_needle, query, ground_truth = generate_needle_haystack_problem(
                        corpus, args.context_length, args.depth
                    )

                    logger.info(f"Generated problem with {len(context_with_needle.split())} words")
                    logger.info(f"Query: {query}")
                    logger.info(f"Ground truth: {ground_truth}")

                    # Get prediction from agent
                    import time
                    start_time = time.time()
                    result = agent.process(context_with_needle, query, extraction_func=extract_needle_answer)
                    latency = time.time() - start_time

                    predicted_answer = result['content']
                    token_stats.append(result.get('token_usage', {}))
                    latency_stats.append(latency)

                    logger.info(f"Predicted answer: {predicted_answer}")
                    logger.info(f"Latency: {latency:.2f}s")

                    # Evaluate accuracy
                    correct = check_needle_answer_match(predicted_answer, ground_truth)
                    accuracy_results.append(int(correct))

                    logger.info(f"{'✓ CORRECT' if correct else '✗ INCORRECT'}")

                    # Log individual run to W&B
                    # Use commit=False to batch multiple runs within the same step
                    # This prevents fragmenting data across thousands of steps
                    wandb.log({
                        "experiment_count": experiment_count,
                        "individual/chunk_size": chunk_size,
                        "individual/run_idx": run_idx,
                        "individual/correct": int(correct),
                        "individual/latency": latency,
                        "individual/avg_completion_tokens": result.get('token_usage', {}).get('avg_completion_tokens', 0),
                        "individual/max_completion_tokens": result.get('token_usage', {}).get('max_completion_tokens', 0),
                        "individual/avg_prompt_tokens": result.get('token_usage', {}).get('avg_prompt_tokens', 0),
                        "individual/max_prompt_tokens": result.get('token_usage', {}).get('max_prompt_tokens', 0),
                    }, commit=False, step=experiment_count)

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

            # Calculate statistics for this chunk size
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
            logger.info(f"RESULTS: Chunk Size={chunk_size}")
            logger.info(f"Accuracy: {avg_accuracy:.3f} ± {se_accuracy:.3f} (max: {max_accuracy:.3f})")
            logger.info(f"Avg Latency: {avg_latency:.2f}s")
            logger.info(f"{'='*70}\n")

            # Calculate average token statistics
            if token_stats:
                completion_tokens_list = [stats.get('avg_completion_tokens', 0) for stats in token_stats]
                prompt_tokens_list = [stats.get('avg_prompt_tokens', 0) for stats in token_stats]

                avg_completion_tokens = mean(completion_tokens_list)
                max_completion_tokens = max([stats.get('max_completion_tokens', 0) for stats in token_stats])
                avg_prompt_tokens = mean(prompt_tokens_list)
                max_prompt_tokens = max([stats.get('max_prompt_tokens', 0) for stats in token_stats])

                # Calculate standard error for token metrics
                if n_runs > 1:
                    std_completion_tokens = stdev(completion_tokens_list)
                    se_completion_tokens = std_completion_tokens / math.sqrt(n_runs)
                    std_prompt_tokens = stdev(prompt_tokens_list)
                    se_prompt_tokens = std_prompt_tokens / math.sqrt(n_runs)
                else:
                    std_completion_tokens = se_completion_tokens = 0.0
                    std_prompt_tokens = se_prompt_tokens = 0.0
            else:
                # Fallback if token_stats is empty (shouldn't happen, but be safe)
                avg_completion_tokens = 0
                max_completion_tokens = 0
                avg_prompt_tokens = 0
                max_prompt_tokens = 0
                std_completion_tokens = 0
                se_completion_tokens = 0
                std_prompt_tokens = 0
                se_prompt_tokens = 0

            # Log aggregated statistics to W&B with commit=True
            # This finalizes the step and ensures data is synced
            # We use experiment_count as the step to maintain chronological order
            wandb.log({
                "experiment_count": experiment_count,
                "agg/chunk_size": chunk_size,
                "agg/avg_accuracy": avg_accuracy,
                "agg/max_accuracy": max_accuracy,
                "agg/std_accuracy": std_accuracy,
                "agg/se_accuracy": se_accuracy,
                "agg/avg_latency": avg_latency,
                "agg/avg_completion_tokens": avg_completion_tokens,
                "agg/max_completion_tokens": max_completion_tokens,
                "agg/std_completion_tokens": std_completion_tokens,
                "agg/se_completion_tokens": se_completion_tokens,
                "agg/avg_prompt_tokens": avg_prompt_tokens,
                "agg/max_prompt_tokens": max_prompt_tokens,
                "agg/std_prompt_tokens": std_prompt_tokens,
                "agg/se_prompt_tokens": se_prompt_tokens,
            }, commit=True, step=experiment_count)

            # Store results for final table
            results_table.append([
                chunk_size,
                f"{avg_accuracy:.3f} ± {se_accuracy:.3f}",
                f"{avg_completion_tokens:.1f}",
                f"{avg_prompt_tokens:.1f}",
                f"{avg_latency:.2f}s"
            ])

    except KeyboardInterrupt:
        logger.warning("\n" + "="*70)
        logger.warning("EXPERIMENT INTERRUPTED BY USER")
        logger.warning(f"Completed {experiment_count}/{total_experiments} experiments")
        logger.warning("="*70)
        raise

    except Exception as e:
        logger.error("\n" + "="*70)
        logger.error("EXPERIMENT FAILED WITH ERROR")
        logger.error(f"Error: {str(e)}")
        logger.error(f"Completed {experiment_count}/{total_experiments} experiments")
        logger.error("="*70)
        raise

    finally:
        # Print final results table
        print("\n" + "="*70)
        print("FINAL RESULTS: CHUNK SIZE ABLATION")
        print("="*70)
        print(tabulate.tabulate(
            results_table,
            headers=["Chunk Size", "Accuracy", "Avg Completion Tokens", "Avg Prompt Tokens", "Avg Latency"],
            tablefmt="grid"
        ))
        print("="*70 + "\n")

        # ALWAYS call wandb.finish() to ensure data is synced, even if script crashes
        logger.info("\n" + "="*70)
        logger.info("EXPERIMENT COMPLETE")
        logger.info(f"Total experiments run: {experiment_count}")
        logger.info(f"Results logged to W&B project: coa-chunk-ablation")
        logger.info("Finalizing W&B sync...")
        logger.info("="*70)

        wandb.finish()


if __name__ == "__main__":
    main()
