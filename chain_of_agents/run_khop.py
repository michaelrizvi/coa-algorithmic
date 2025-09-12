import argparse
import random
import logging
import wandb
from statistics import mean, stdev
import math

from logger import setup_logger
from main import MajorityVotingAgents, IterativeQueryAgents
from utils import *
import tabulate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_type", choices=["MajorityVotingAgents", "IterativeQueryAgents"], default="IterativeQueryAgents", help="Type of agent to use")
    parser.add_argument("--num_agents", type=int, default=3, help="Number of agents to use in MajVote setup")
    parser.add_argument("--model_type", type=str, default="meta-llama/Llama-3.3-70B-Instruct-Turbo", help="Model type to use for agents")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max tokens for each agent")
    parser.add_argument("--num_runs", type=int, default=5, help="Number of runs to perform")
    parser.add_argument("--min_hops", type=int, default=20, help="Minimum number of hops")
    parser.add_argument("--max_hops", type=int, default=20, help="Maximum number of hops") 
    parser.add_argument("--num_facts", type=int, default=100, help="Number of facts to include in each problem")
    parser.add_argument("--facts_per_worker", type=int, default=20, help="Number of facts per worker (for IterativeQueryAgents)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--step", type=int, default=2, help="Step size for hops")
    args = parser.parse_args()

    # Create a nice table showing all arguments
    args_table = []
    for arg, value in vars(args).items():
        args_table.append([arg, value])
        
    print("\n" + "="*50)
    print("K-HOP EXPERIMENT CONFIGURATION")
    print("="*50)
    print(tabulate.tabulate(args_table, headers=["Argument", "Value"], tablefmt="grid"))
    print("="*50 + "\n")
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    if args.agent_type == "MajorityVotingAgents":
        run_name = f"khop_{args.agent_type}_agents{args.num_agents}_hops{args.min_hops}-{args.max_hops}_facts{args.num_facts}"
    elif args.agent_type == "IterativeQueryAgents":
        run_name = f"khop_{args.agent_type}_factsperworker{args.facts_per_worker}_hops{args.min_hops}-{args.max_hops}_facts{args.num_facts}"
    wandb.init(project="coa-khop-eval", config=vars(args), name=run_name, reinit=True)
    logger = setup_logger()

    # Initialize agent
    if args.agent_type == "MajorityVotingAgents":
        prompt = get_khop_majority_vote_prompt()
        agent = MajorityVotingAgents(
            num_agents=args.num_agents,
            model=args.model_type,
            max_tokens=args.max_tokens,
            prompt=prompt,
        )
    elif args.agent_type == "IterativeQueryAgents":
        agent = IterativeQueryAgents(
            worker_model=args.model_type,
            manager_model=args.model_type,
            facts_per_worker=args.facts_per_worker,
            max_tokens_worker=args.max_tokens,
            max_tokens_manager=args.max_tokens,
            num_hops=args.max_hops  # Set to max hops for flexibility
        )

    # Get entities and relations
    entities = get_khop_entities()
    relations = get_khop_relations()
    logger.info(f"Using {len(entities)} entities and {len(relations)} relations")

    # Run experiments for different numbers of hops
    for num_hops in range(args.min_hops, args.max_hops + 1, args.step):
        accuracy_results = []
        token_stats = []
        
        for run_idx in range(args.num_runs):
            # Generate K-hop problem
            try:
                facts, query, ground_truth = generate_khop_problem(num_hops, args.num_facts, entities, relations)
                
                logger.info(f"Run {run_idx+1}/{args.num_runs} - Hops: {num_hops}, Facts: {args.num_facts}")
                logger.info(f"Query: {query}")
                logger.info(f"Ground truth: {ground_truth}")
                logger.info(f"Facts: {facts[:200]}...")  # Log first 200 chars of facts
                
                # Get prediction from agent
                input_text = facts
                if args.agent_type == "MajorityVotingAgents":
                    result = agent.process(input_text, query, extraction_func=extract_khop_answer)
                elif args.agent_type == "IterativeQueryAgents":
                    # Update the agent's num_hops for this specific problem
                    agent.num_hops = num_hops
                    result = agent.process(input_text, query, extraction_func=extract_khop_answer)
                
                predicted_answer = result['content']  # Already extracted by the process method
                token_stats.append(result['token_usage'])
                logger.info(f"Predicted answer: {predicted_answer}")
                
                # Evaluate accuracy (exact match, case-insensitive)
                correct = int(predicted_answer.lower() == ground_truth.lower())
                accuracy_results.append(correct)
                
                logger.info(f"Correct: {correct}")
                
            except Exception as e:
                logger.error(f"Error processing run {run_idx+1}: {str(e)}")
                accuracy_results.append(0)
                # Add dummy token stats for failed runs
                token_stats.append({
                    'avg_completion_tokens': 0,
                    'max_completion_tokens': 0,
                    'avg_prompt_tokens': 0,
                    'max_prompt_tokens': 0
                })
        
        # Calculate statistics for this hop count
        avg_accuracy = mean(accuracy_results)
        max_accuracy = max(accuracy_results)
        
        # Calculate standard error (SE = std_dev / sqrt(n))
        n_runs = len(accuracy_results)
        if n_runs > 1:
            std_accuracy = stdev(accuracy_results)
            se_accuracy = std_accuracy / math.sqrt(n_runs)
        else:
            std_accuracy = se_accuracy = 0.0
        
        logger.info(f"Hops={num_hops}, AvgAccuracy={avg_accuracy:.3f}±{se_accuracy:.3f}")
        logger.info(f"Hops={num_hops}, MaxAccuracy={max_accuracy:.3f}")
        
        # Calculate average token statistics for this number of hops
        if token_stats:
            avg_completion_tokens = mean([stats['avg_completion_tokens'] for stats in token_stats])
            max_completion_tokens = max([stats['max_completion_tokens'] for stats in token_stats])
            avg_prompt_tokens = mean([stats['avg_prompt_tokens'] for stats in token_stats])
            max_prompt_tokens = max([stats['max_prompt_tokens'] for stats in token_stats])
            
            logger.info(f"Hops={num_hops}, AvgCompletionTokens={avg_completion_tokens:.2f}")
            logger.info(f"Hops={num_hops}, MaxCompletionTokens={max_completion_tokens:.2f}")
            logger.info(f"Hops={num_hops}, AvgPromptTokens={avg_prompt_tokens:.2f}")
            logger.info(f"Hops={num_hops}, MaxPromptTokens={max_prompt_tokens:.2f}")
            
            wandb.log({
                "avg_accuracy": avg_accuracy,
                "max_accuracy": max_accuracy,
                "std_accuracy": std_accuracy,
                "se_accuracy": se_accuracy,
                "avg_completion_tokens": avg_completion_tokens,
                "max_completion_tokens": max_completion_tokens,
                "avg_prompt_tokens": avg_prompt_tokens,
                "max_prompt_tokens": max_prompt_tokens,
                "num_hops": num_hops
            })
        else:
            wandb.log({
                "avg_accuracy": avg_accuracy,
                "max_accuracy": max_accuracy,
                "std_accuracy": std_accuracy,
                "se_accuracy": se_accuracy,
                "num_hops": num_hops
            })


if __name__ == "__main__":
    main()