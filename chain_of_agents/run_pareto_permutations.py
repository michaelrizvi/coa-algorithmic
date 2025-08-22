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
    parser.add_argument("--model_type", type=str, default="lgai/exaone-3-5-32b-instruct", help="Model type to use for agents")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max tokens for each agent")
    parser.add_argument("--num_runs", type=int, default=10, help="Number of runs to perform")
    parser.add_argument("--num_elements", type=int, default=5, help="Number of elements in permutation")
    parser.add_argument("--num_swaps", type=int, default=10, help="Number of swaps in permutation problem")
    parser.add_argument("--min_branching_factor", type=int, default=2, help="Minimum branching factor")
    parser.add_argument("--max_branching_factor", type=int, default=4, help="Maximum branching factor")
    parser.add_argument("--step", type=int, default=2, help="Step size for branching factor")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Create a nice table showing all arguments
    args_table = []
    for arg, value in vars(args).items():
        args_table.append([arg, value])
        
    print("\n" + "="*50)
    print("PARETO PERMUTATION EXPERIMENT CONFIGURATION")
    print("="*50)
    print(tabulate.tabulate(args_table, headers=["Argument", "Value"], tablefmt="grid"))
    print("="*50 + "\n")
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    run_name = f"pareto_permutation_elements{args.num_elements}_swaps{args.num_swaps}_b{args.min_branching_factor}-{args.max_branching_factor}"
    wandb.init(project="coa-pareto-permutation-eval", config=vars(args), name=run_name, reinit=True)
    logger = setup_logger()

    # Loop over branching factors
    for branching_factor in range(args.min_branching_factor, args.max_branching_factor + 1, args.step):
        # Initialize PrefixSumAgents for this branching factor
        worker_prompt, manager_prompt = get_permutation_prefix_sum_prompts(b=branching_factor)
        agent = PrefixSumAgents(
            worker_model=args.model_type,
            manager_model=args.model_type,
            max_tokens_worker=args.max_tokens,
            max_tokens_manager=args.max_tokens,
            worker_prompt=worker_prompt,
            manager_prompt=manager_prompt,
            branching_factor=branching_factor,
        )

        exact_match_results = []
        element_accuracy_results = []
        token_stats = []
        
        for run_idx in range(args.num_runs):
            # Generate permutation problem
            swap_sequence, true_positions = generate_permutation_problem(n=args.num_elements, num_swaps=args.num_swaps)
            query = "What is the final position of each ball after all swaps?"
            
            logger.info(f"BranchingFactor={branching_factor}, Run={run_idx+1}, Elements: {args.num_elements}, Swaps: {args.num_swaps}")
            logger.info(f"BranchingFactor={branching_factor}, Run={run_idx+1}, Swap sequence: {swap_sequence}")
            logger.info(f"BranchingFactor={branching_factor}, Run={run_idx+1}, True positions: {true_positions}")

            try:
                result = agent.hierarchical_process(swap_sequence, query, extraction_func=extract_position_dict)
                
                # Handle case where result['content'] is None
                if result['content'] is None:
                    predicted_positions = None
                    logger.warning(f"BranchingFactor={branching_factor}, Run={run_idx+1}, Result content is None")
                else:
                    pred = result['content']
                    logger.info(f"BranchingFactor={branching_factor}, Run={run_idx+1}, Raw prediction: {pred}")
                    
                    # Parse prediction
                    if isinstance(pred, str):
                        if pred.startswith("{") and pred.endswith("}"):
                            predicted_positions = parse_position_dict(pred)
                        else:
                            # Try to extract dict from the prediction
                            extracted_dict = extract_position_dict(pred)
                            predicted_positions = parse_position_dict(extracted_dict) if extracted_dict else None
                    else:
                        predicted_positions = pred
                    
                    logger.info(f"BranchingFactor={branching_factor}, Run={run_idx+1}, Parsed prediction: {predicted_positions}")
                
                token_stats.append(result['token_usage'])
                
                # Add debugging for dictionary composition failures
                if predicted_positions is None:
                    logger.warning(f"BranchingFactor={branching_factor}, Run={run_idx+1}, FAILED PARSING: Could not extract dictionary from: {str(pred)[:100] if pred else 'None'}...")
                elif not isinstance(predicted_positions, dict):
                    logger.warning(f"BranchingFactor={branching_factor}, Run={run_idx+1}, FAILED PARSING: Extracted non-dict: {type(predicted_positions)} = {predicted_positions}")
                
                # Evaluate accuracy
                exact_match, element_accuracy = evaluate_permutation_accuracy(predicted_positions, true_positions)
                exact_match_results.append(int(exact_match))
                element_accuracy_results.append(element_accuracy)
                
                logger.info(f"BranchingFactor={branching_factor}, Run={run_idx+1}, Exact match: {exact_match}, Element accuracy: {element_accuracy:.3f}")
                
            except Exception as e:
                logger.error(f"BranchingFactor={branching_factor}, Run={run_idx+1}, Error processing: {str(e)}")
                exact_match_results.append(0)
                element_accuracy_results.append(0.0)
                # Add dummy token stats for failed runs
                token_stats.append({
                    'avg_completion_tokens': 0,
                    'max_completion_tokens': 0,
                    'avg_prompt_tokens': 0,
                    'max_prompt_tokens': 0,
                    'mean_completion_tokens_per_agent': 0,
                    'max_completion_tokens_per_agent': 0,
                    'mean_prompt_tokens_per_agent': 0,
                    'max_prompt_tokens_per_agent': 0
                })
        
        # Calculate statistics for this branching factor
        avg_exact_match = mean(exact_match_results)
        avg_element_accuracy = mean(element_accuracy_results)
        max_exact_match = max(exact_match_results)
        max_element_accuracy = max(element_accuracy_results)
        
        # Calculate standard error (SE = std_dev / sqrt(n))
        n_runs = len(exact_match_results)
        if n_runs > 1:
            std_exact_match = stdev(exact_match_results)
            std_element_accuracy = stdev(element_accuracy_results)
            se_exact_match = std_exact_match / math.sqrt(n_runs)
            se_element_accuracy = std_element_accuracy / math.sqrt(n_runs)
        else:
            std_exact_match = se_exact_match = 0.0
            std_element_accuracy = se_element_accuracy = 0.0
        
        logger.info(f"BranchingFactor={branching_factor}, AvgExactMatch={avg_exact_match:.3f}±{se_exact_match:.3f}")
        logger.info(f"BranchingFactor={branching_factor}, AvgElementAccuracy={avg_element_accuracy:.3f}±{se_element_accuracy:.3f}")
        logger.info(f"BranchingFactor={branching_factor}, MaxExactMatch={max_exact_match:.3f}, MaxElementAccuracy={max_element_accuracy:.3f}")
        
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
                "avg_exact_match": avg_exact_match,
                "avg_element_accuracy": avg_element_accuracy,
                "max_exact_match": max_exact_match,
                "max_element_accuracy": max_element_accuracy,
                "std_exact_match": std_exact_match,
                "std_element_accuracy": std_element_accuracy,
                "se_exact_match": se_exact_match,
                "se_element_accuracy": se_element_accuracy,
                "avg_completion_tokens": avg_completion_tokens,
                "max_completion_tokens": max_completion_tokens,
                "avg_prompt_tokens": avg_prompt_tokens,
                "max_prompt_tokens": max_prompt_tokens,
                "mean_completion_tokens_per_agent": mean_completion_tokens_per_agent,
                "max_completion_tokens_per_agent": max_completion_tokens_per_agent,
                "mean_prompt_tokens_per_agent": mean_prompt_tokens_per_agent,
                "max_prompt_tokens_per_agent": max_prompt_tokens_per_agent,
                "branching_factor": branching_factor,
                "num_elements": args.num_elements,
                "num_swaps": args.num_swaps
            })
        else:
            wandb.log({
                "avg_exact_match": avg_exact_match,
                "avg_element_accuracy": avg_element_accuracy,
                "max_exact_match": max_exact_match,
                "max_element_accuracy": max_element_accuracy,
                "std_exact_match": std_exact_match,
                "std_element_accuracy": std_element_accuracy,
                "se_exact_match": se_exact_match,
                "se_element_accuracy": se_element_accuracy,
                "branching_factor": branching_factor,
                "num_elements": args.num_elements,
                "num_swaps": args.num_swaps
            })


if __name__ == "__main__":
    main()