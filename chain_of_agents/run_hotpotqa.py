import argparse
import random
import logging
import wandb
from statistics import mean, stdev
import math

from logger import setup_logger
from main import BridgeQueryAgents, MajorityVotingAgents
from utils import (
    load_hotpotqa_data,
    extract_hotpotqa_answer,
    hotpotqa_exact_match,
    hotpotqa_f1_score,
    get_bridge_query_prompts,
    get_hotpotqa_majority_vote_prompt
)
import tabulate


def main():
    parser = argparse.ArgumentParser(description="Benchmark agents on HotPotQA bridge questions")
    parser.add_argument("--agent_type", choices=["MajorityVotingAgents", "BridgeQueryAgents"],
                       default="BridgeQueryAgents", help="Type of agent to use")
    parser.add_argument("--model_type", type=str, default="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                       help="Model type to use for agents")
    parser.add_argument("--num_examples", type=int, default=2000,
                       help="Number of examples from test set to evaluate")
    parser.add_argument("--use_gold_only", action="store_true",
                       help="Use only supporting paragraphs (gold context)")
    parser.add_argument("--chunk_size", type=int, default=500,
                       help="Words per worker chunk (BridgeQueryAgents only)")
    parser.add_argument("--num_agents", type=int, default=3,
                       help="Number of agents (MajorityVotingAgents only)")
    parser.add_argument("--max_tokens", type=int, default=2048,
                       help="Max tokens for each agent")
    parser.add_argument("--num_runs", type=int, default=5,
                       help="Number of repetitions per example for variance estimation")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    args = parser.parse_args()

    # Create a nice table showing all arguments
    args_table = []
    for arg, value in vars(args).items():
        args_table.append([arg, value])

    print("\n" + "="*50)
    print("HOTPOTQA BENCHMARK CONFIGURATION")
    print("="*50)
    print(tabulate.tabulate(args_table, headers=["Argument", "Value"], tablefmt="grid"))
    print("="*50 + "\n")

    # Set random seed for reproducibility
    random.seed(args.seed)

    # Create wandb run name
    if args.agent_type == "MajorityVotingAgents":
        run_name = f"hotpotqa_{args.agent_type}_agents{args.num_agents}_examples{args.num_examples}_gold{args.use_gold_only}"
    elif args.agent_type == "BridgeQueryAgents":
        run_name = f"hotpotqa_{args.agent_type}_chunk{args.chunk_size}_examples{args.num_examples}_gold{args.use_gold_only}"

    # Initialize W&B with settings optimized for SLURM/cluster environments
    wandb.init(
        project="coa-hotpotqa-eval",
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
    wandb.define_metric("run_count")  # Our global step counter
    wandb.define_metric("individual/*", step_metric="run_count")  # Individual run metrics
    wandb.define_metric("agg/*", step_metric="run_count")  # Aggregated metrics per example
    wandb.define_metric("final/*")  # Final summary metrics (no step)

    logger = setup_logger(enable_wandb=False)  # Disable wandb logging to prevent excessive API calls

    # Load HotPotQA data
    logger.info("Loading HotPotQA dataset...")
    dataset = load_hotpotqa_data(subset_size=args.num_examples, use_gold_only=args.use_gold_only)
    logger.info(f"Loaded {len(dataset)} bridge questions")

    # Initialize agent
    if args.agent_type == "BridgeQueryAgents":
        logger.info("Initializing BridgeQueryAgents...")
        prompts = get_bridge_query_prompts()
        agent = BridgeQueryAgents(
            worker_model=args.model_type,
            manager_model=args.model_type,
            chunk_size=args.chunk_size,
            max_tokens_worker=args.max_tokens,
            max_tokens_manager=args.max_tokens,
            worker_prompt=prompts[2],
            query_splitter_prompt=prompts[0],
            query_updater_prompt=prompts[1]
        )
    elif args.agent_type == "MajorityVotingAgents":
        logger.info("Initializing MajorityVotingAgents...")
        prompt = get_hotpotqa_majority_vote_prompt()
        agent = MajorityVotingAgents(
            num_agents=args.num_agents,
            model=args.model_type,
            max_tokens=args.max_tokens,
            prompt=prompt
        )

    # Run experiments
    overall_em_scores = []
    overall_f1_scores = []
    overall_token_stats = []
    run_count = 0  # Global counter for step management

    try:
        for example_idx, example in enumerate(dataset):
            logger.info(f"\n{'='*60}")
            logger.info(f"Example {example_idx+1}/{len(dataset)}")
            logger.info(f"{'='*60}")

            question = example["question"]
            context = example["context"]
            ground_truth = example["answer"]

            logger.info(f"Question: {question}")
            logger.info(f"Ground truth: {ground_truth}")
            logger.info(f"Context length: {len(context.split())} words")

            # Run multiple times for variance estimation
            run_em_scores = []
            run_f1_scores = []
            run_token_stats = []

            for run_idx in range(args.num_runs):
                run_count += 1  # Increment global step counter
                logger.info(f"\nRun {run_idx+1}/{args.num_runs} (global run #{run_count})")

                try:
                    # Get prediction from agent
                    result = agent.process(context, question, extraction_func=extract_hotpotqa_answer)

                    predicted_answer = result['content']
                    token_usage = result['token_usage']

                    logger.info(f"Predicted answer: {predicted_answer}")

                    # Evaluate using official HotPotQA metrics
                    em_score = hotpotqa_exact_match(predicted_answer, ground_truth)
                    f1 = hotpotqa_f1_score(predicted_answer, ground_truth)

                    logger.info(f"EM: {int(em_score)}, F1: {f1:.3f}")
                    logger.info(f"Token usage - Avg completion: {token_usage['avg_completion_tokens']:.2f}, "
                              f"Max completion: {token_usage['max_completion_tokens']}")

                    run_em_scores.append(int(em_score))
                    run_f1_scores.append(f1)
                    run_token_stats.append(token_usage)

                    # Log individual run to wandb
                    # Use commit=False to batch multiple runs within the same step
                    # Don't log full text (predicted/ground_truth/question) to reduce bandwidth
                    wandb.log({
                        "run_count": run_count,
                        "individual/example_id": example_idx,
                        "individual/run_id": run_idx,
                        "individual/em": int(em_score),
                        "individual/f1": f1,
                        "individual/avg_completion_tokens": token_usage['avg_completion_tokens'],
                        "individual/max_completion_tokens": token_usage['max_completion_tokens'],
                        "individual/avg_prompt_tokens": token_usage['avg_prompt_tokens'],
                        "individual/max_prompt_tokens": token_usage['max_prompt_tokens']
                    }, commit=False, step=run_count)

                except Exception as e:
                    logger.error(f"Error processing example {example_idx}, run {run_idx}: {str(e)}")
                    run_em_scores.append(0)
                    run_f1_scores.append(0.0)
                    # Add dummy token stats for failed runs
                    run_token_stats.append({
                        'avg_completion_tokens': 0,
                        'max_completion_tokens': 0,
                        'avg_prompt_tokens': 0,
                        'max_prompt_tokens': 0
                    })

            # Calculate statistics for this example across runs
            avg_em = mean(run_em_scores)
            avg_f1 = mean(run_f1_scores)

            # Calculate standard error (SE = std_dev / sqrt(n))
            if len(run_em_scores) > 1:
                std_em = stdev(run_em_scores)
                se_em = std_em / math.sqrt(len(run_em_scores))
                std_f1 = stdev(run_f1_scores)
                se_f1 = std_f1 / math.sqrt(len(run_f1_scores))
            else:
                std_em = 0.0
                se_em = 0.0
                std_f1 = 0.0
                se_f1 = 0.0

            # Average token stats across runs
            avg_completion_tokens = mean([stats['avg_completion_tokens'] for stats in run_token_stats])
            max_completion_tokens = max([stats['max_completion_tokens'] for stats in run_token_stats])
            avg_prompt_tokens = mean([stats['avg_prompt_tokens'] for stats in run_token_stats])
            max_prompt_tokens = max([stats['max_prompt_tokens'] for stats in run_token_stats])

            logger.info(f"\nExample {example_idx} Summary:")
            logger.info(f"Avg EM: {avg_em:.3f} ± {se_em:.3f}")
            logger.info(f"Avg F1: {avg_f1:.3f} ± {se_f1:.3f}")
            logger.info(f"Avg Completion Tokens: {avg_completion_tokens:.2f}")

            # Store for overall statistics
            overall_em_scores.extend(run_em_scores)
            overall_f1_scores.extend(run_f1_scores)
            overall_token_stats.extend(run_token_stats)

            # Log aggregated example stats to wandb with commit=True
            # This finalizes the step and ensures data is synced
            wandb.log({
                "run_count": run_count,
                "agg/example_id": example_idx,
                "agg/avg_em": avg_em,
                "agg/se_em": se_em,
                "agg/std_em": std_em,
                "agg/avg_f1": avg_f1,
                "agg/se_f1": se_f1,
                "agg/std_f1": std_f1,
                "agg/avg_completion_tokens": avg_completion_tokens,
                "agg/max_completion_tokens": max_completion_tokens,
                "agg/avg_prompt_tokens": avg_prompt_tokens,
                "agg/max_prompt_tokens": max_prompt_tokens,
                "agg/num_runs": len(run_em_scores)
            }, commit=True, step=run_count)

    except KeyboardInterrupt:
        logger.warning("\n" + "="*60)
        logger.warning("EXPERIMENT INTERRUPTED BY USER")
        logger.warning(f"Completed {len(overall_em_scores)}/{len(dataset) * args.num_runs} runs")
        logger.warning("="*60)
        raise

    except Exception as e:
        logger.error("\n" + "="*60)
        logger.error("EXPERIMENT FAILED WITH ERROR")
        logger.error(f"Error: {str(e)}")
        logger.error(f"Completed {len(overall_em_scores)}/{len(dataset) * args.num_runs} runs")
        logger.error("="*60)
        raise

    finally:
        # ALWAYS calculate and log final statistics, even if interrupted
        # Calculate overall statistics for EM
        overall_avg_em = mean(overall_em_scores) if overall_em_scores else 0.0

        if len(overall_em_scores) > 1:
            overall_std_em = stdev(overall_em_scores)
            overall_se_em = overall_std_em / math.sqrt(len(overall_em_scores))
        else:
            overall_std_em = 0.0
            overall_se_em = 0.0

        # Calculate overall statistics for F1
        overall_avg_f1 = mean(overall_f1_scores) if overall_f1_scores else 0.0

        if len(overall_f1_scores) > 1:
            overall_std_f1 = stdev(overall_f1_scores)
            overall_se_f1 = overall_std_f1 / math.sqrt(len(overall_f1_scores))
        else:
            overall_std_f1 = 0.0
            overall_se_f1 = 0.0

        overall_avg_completion_tokens = mean([stats['avg_completion_tokens'] for stats in overall_token_stats]) if overall_token_stats else 0.0
        overall_max_completion_tokens = max([stats['max_completion_tokens'] for stats in overall_token_stats]) if overall_token_stats else 0.0
        overall_avg_prompt_tokens = mean([stats['avg_prompt_tokens'] for stats in overall_token_stats]) if overall_token_stats else 0.0
        overall_max_prompt_tokens = max([stats['max_prompt_tokens'] for stats in overall_token_stats]) if overall_token_stats else 0.0

        # Print final summary
        print("\n" + "="*60)
        print("FINAL RESULTS")
        print("="*60)

        summary_table = [
            ["Total Examples", len(dataset)],
            ["Runs per Example", args.num_runs],
            ["Total Runs", len(overall_em_scores)],
            ["Overall EM", f"{overall_avg_em:.3f} ± {overall_se_em:.3f}"],
            ["Overall F1", f"{overall_avg_f1:.3f} ± {overall_se_f1:.3f}"],
            ["Avg Completion Tokens", f"{overall_avg_completion_tokens:.2f}"],
            ["Max Completion Tokens", f"{overall_max_completion_tokens}"],
            ["Avg Prompt Tokens", f"{overall_avg_prompt_tokens:.2f}"],
            ["Max Prompt Tokens", f"{overall_max_prompt_tokens}"]
        ]

        print(tabulate.tabulate(summary_table, headers=["Metric", "Value"], tablefmt="grid"))
        print("="*60 + "\n")

        # Log final summary to wandb with final/ prefix (no step)
        wandb.log({
            "final/overall_em": overall_avg_em,
            "final/overall_se_em": overall_se_em,
            "final/overall_std_em": overall_std_em,
            "final/overall_f1": overall_avg_f1,
            "final/overall_se_f1": overall_se_f1,
            "final/overall_std_f1": overall_std_f1,
            "final/avg_completion_tokens": overall_avg_completion_tokens,
            "final/max_completion_tokens": overall_max_completion_tokens,
            "final/avg_prompt_tokens": overall_avg_prompt_tokens,
            "final/max_prompt_tokens": overall_max_prompt_tokens,
            "final/total_runs": len(overall_em_scores),
            "final/total_examples": len(dataset)
        })

        logger.info("Finalizing W&B sync...")
        wandb.finish()
        logger.info("Experiment complete!")


if __name__ == "__main__":
    main()
