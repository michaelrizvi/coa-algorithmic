import argparse
import random
import logging
import wandb
from statistics import mean, stdev
import math

from logger import setup_logger
from main import IterativeQueryAgents
from utils import *
import tabulate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="meta-llama/Llama-3.3-70B-Instruct-Turbo", help="Model type to use for agents")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max tokens for each agent")
    parser.add_argument("--num_runs", type=int, default=5, help="Number of runs to perform")
    parser.add_argument("--nb_hops", type=int, default=5, help="Number of hops for the K-hop problem")
    parser.add_argument("--nb_facts", type=int, default=500, help="Number of facts to include in each problem")
    parser.add_argument("--min_facts_per_worker", type=int, default=20, help="Minimum number of facts per worker")
    parser.add_argument("--max_facts_per_worker", type=int, default=100, help="Maximum number of facts per worker")
    parser.add_argument("--step", type=int, default=20, help="Step size for facts_per_worker iteration")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Create a nice table showing all arguments
    args_table = []
    for arg, value in vars(args).items():
        args_table.append([arg, value])
        
    print("\n" + "="*50)
    print("K-HOP PARETO EXPERIMENT CONFIGURATION")
    print("="*50)
    print(tabulate.tabulate(args_table, headers=["Argument", "Value"], tablefmt="grid"))
    print("="*50 + "\n")
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    run_name = f"khop_pareto_factsperworker{args.min_facts_per_worker}-{args.max_facts_per_worker}_hops{args.nb_hops}_facts{args.nb_facts}"
    wandb.init(project="coa-khop-pareto-eval", config=vars(args), name=run_name, reinit=True)
    logger = setup_logger()

    # Get entities and relations
    entities = get_khop_entities()
    relations = get_khop_relations()
    logger.info(f"Using {len(entities)} entities and {len(relations)} relations")

    # Loop over facts_per_worker values
    for facts_per_worker in range(args.min_facts_per_worker, args.max_facts_per_worker + 1, args.step):
        # Initialize IterativeQueryAgents for this facts_per_worker value
        agent = IterativeQueryAgents(
            worker_model=args.model_type,
            manager_model=args.model_type,
            facts_per_worker=facts_per_worker,
            max_tokens_worker=args.max_tokens,
            max_tokens_manager=args.max_tokens,
            num_hops=args.nb_hops
        )

        accuracy_results = []
        token_stats = []
        
        for run_idx in range(args.num_runs):
            # Generate K-hop problem
            try:
                facts, query, ground_truth = generate_khop_problem(args.nb_hops, args.nb_facts, entities, relations)
                
                logger.info(f"FactsPerWorker={facts_per_worker}, Run {run_idx+1}/{args.num_runs} - Hops: {args.nb_hops}, Facts: {args.nb_facts}")
                logger.info(f"FactsPerWorker={facts_per_worker}, Query: {query}")
                logger.info(f"FactsPerWorker={facts_per_worker}, Ground truth: {ground_truth}")
                logger.info(f"FactsPerWorker={facts_per_worker}, Facts: {facts[:200]}...")  # Log first 200 chars of facts
                
                # Get prediction from agent
                input_text = facts
                result = agent.process(input_text, query, extraction_func=extract_khop_answer)
                
                predicted_answer = result['content']  # Already extracted by the process method
                token_stats.append(result['token_usage'])
                logger.info(f"FactsPerWorker={facts_per_worker}, Predicted answer: {predicted_answer}")
                
                # Evaluate accuracy (exact match, case-insensitive)
                correct = int(predicted_answer.lower() == ground_truth.lower())
                accuracy_results.append(correct)
                
                logger.info(f"FactsPerWorker={facts_per_worker}, Correct: {correct}")
                
            except Exception as e:
                logger.error(f"FactsPerWorker={facts_per_worker}, Error processing run {run_idx+1}: {str(e)}")
                accuracy_results.append(0)
                # Add dummy token stats for failed runs
                token_stats.append({
                    'avg_completion_tokens': 0,
                    'max_completion_tokens': 0,
                    'avg_prompt_tokens': 0,
                    'max_prompt_tokens': 0
                })
        
        # Calculate statistics for this facts_per_worker value
        avg_accuracy = mean(accuracy_results)
        max_accuracy = max(accuracy_results)
        
        # Calculate standard error (SE = std_dev / sqrt(n))
        n_runs = len(accuracy_results)
        if n_runs > 1:
            std_accuracy = stdev(accuracy_results)
            se_accuracy = std_accuracy / math.sqrt(n_runs)
        else:
            std_accuracy = se_accuracy = 0.0
        
        logger.info(f"FactsPerWorker={facts_per_worker}, AvgAccuracy={avg_accuracy:.3f}±{se_accuracy:.3f}")
        logger.info(f"FactsPerWorker={facts_per_worker}, MaxAccuracy={max_accuracy:.3f}")
        
        # Calculate average token statistics for this facts_per_worker value
        if token_stats:
            avg_completion_tokens = mean([stats['avg_completion_tokens'] for stats in token_stats])
            max_completion_tokens = max([stats['max_completion_tokens'] for stats in token_stats])
            avg_prompt_tokens = mean([stats['avg_prompt_tokens'] for stats in token_stats])
            max_prompt_tokens = max([stats['max_prompt_tokens'] for stats in token_stats])
            
            logger.info(f"FactsPerWorker={facts_per_worker}, AvgCompletionTokens={avg_completion_tokens:.2f}")
            logger.info(f"FactsPerWorker={facts_per_worker}, MaxCompletionTokens={max_completion_tokens:.2f}")
            logger.info(f"FactsPerWorker={facts_per_worker}, AvgPromptTokens={avg_prompt_tokens:.2f}")
            logger.info(f"FactsPerWorker={facts_per_worker}, MaxPromptTokens={max_prompt_tokens:.2f}")
            
            wandb.log({
                "avg_accuracy": avg_accuracy,
                "max_accuracy": max_accuracy,
                "std_accuracy": std_accuracy,
                "se_accuracy": se_accuracy,
                "avg_completion_tokens": avg_completion_tokens,
                "max_completion_tokens": max_completion_tokens,
                "avg_prompt_tokens": avg_prompt_tokens,
                "max_prompt_tokens": max_prompt_tokens,
                "facts_per_worker": facts_per_worker,
                "nb_hops": args.nb_hops,
                "nb_facts": args.nb_facts
            })
        else:
            wandb.log({
                "avg_accuracy": avg_accuracy,
                "max_accuracy": max_accuracy,
                "std_accuracy": std_accuracy,
                "se_accuracy": se_accuracy,
                "facts_per_worker": facts_per_worker,
                "nb_hops": args.nb_hops,
                "nb_facts": args.nb_facts
            })


if __name__ == "__main__":
    main()