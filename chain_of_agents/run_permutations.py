import argparse
import random
import logging
import wandb
from statistics import mean

from logger import setup_logger
from main import MajorityVotingAgents, ChainOfAgents, PrefixSumAgents
from utils import *
import tabulate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_type", choices=["MajorityVotingAgents", "ChainOfAgents", "PrefixSumAgents"], default="ChainOfAgents", help="Type of agent to use")
    parser.add_argument("--num_agents", type=int, default=3, help="Number of agents to use in MajVote setup")
    parser.add_argument("--model_type", type=str, default="lgai/exaone-3-5-32b-instruct", help="Model type to use for agents")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max tokens for each agent")
    parser.add_argument("--chunk_size", type=int, default=2, help="Chunk size for Chain of Agents (number of swaps)")
    parser.add_argument("--num_runs", type=int, default=10, help="Number of runs to perform")
    parser.add_argument("--num_elements", type=int, default=5, help="Number of elements in permutation")
    parser.add_argument("--min_swaps", type=int, default=4, help="Minimum number of swaps")
    parser.add_argument("--max_swaps", type=int, default=16, help="Maximum number of swaps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--branching_factor", type=int, default=2, help="Branching factor for prefix sum agents")
    args = parser.parse_args()

    # Create a nice table showing all arguments
    args_table = []
    for arg, value in vars(args).items():
        args_table.append([arg, value])
        
    print("\n" + "="*50)
    print("PERMUTATION EXPERIMENT CONFIGURATION")
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
    
    if args.agent_type == "MajorityVotingAgents":
        run_name = f"{name_dict[args.agent_type]}_elements{args.num_elements}_agents{args.num_agents}_swaps{args.min_swaps}-{args.max_swaps}"
    elif args.agent_type == "ChainOfAgents":
        run_name = f"{name_dict[args.agent_type]}_elements{args.num_elements}_chunk{args.chunk_size}_swaps{args.min_swaps}-{args.max_swaps}"
    elif args.agent_type == "PrefixSumAgents":
        run_name = f"{name_dict[args.agent_type]}_elements{args.num_elements}_b{args.branching_factor}_swaps{args.min_swaps}-{args.max_swaps}"
    
    wandb.init(project="coa-permutation-eval", config=vars(args), name=run_name, reinit=True)
    logger = setup_logger()

    # Initialize agents
    if args.agent_type == "MajorityVotingAgents":
        prompt = get_permutation_majority_vote_prompt()
        agent = MajorityVotingAgents(
            num_agents=args.num_agents,
            model=args.model_type,
            max_tokens=args.max_tokens,
            prompt=prompt,
        )
    elif args.agent_type == "ChainOfAgents":
        worker_prompt, manager_prompt = get_permutation_prompts()
        agent = ChainOfAgents(
            worker_model=args.model_type,
            manager_model=args.model_type,
            chunk_size=args.chunk_size,
            worker_prompt=worker_prompt,
            manager_prompt=manager_prompt,
            max_tokens_worker=args.max_tokens,
            use_index_hints=False,
        )
    elif args.agent_type == "PrefixSumAgents":
        worker_prompt, manager_prompt = get_permutation_prefix_sum_prompts(b=args.branching_factor)
        agent = PrefixSumAgents(
            worker_model=args.model_type,
            manager_model=args.model_type,
            max_tokens_worker=args.max_tokens,
            max_tokens_manager=args.max_tokens,
            worker_prompt=worker_prompt,
            manager_prompt=manager_prompt,
            branching_factor=args.branching_factor,
        )

    # Run experiments for different numbers of swaps
    for num_swaps in range(args.min_swaps, args.max_swaps + 1):
        exact_match_results = []
        element_accuracy_results = []
        
        for run_idx in range(args.num_runs):
            # Generate permutation problem
            swap_sequence, true_positions = generate_permutation_problem(n=args.num_elements, num_swaps=num_swaps)
            
            query = "What is the final position of each ball after all swaps?"
            
            logger.info(f"Run {run_idx+1}/{args.num_runs} - Elements: {args.num_elements}, Swaps: {num_swaps}")
            logger.info(f"Swap sequence: {swap_sequence}")
            logger.info(f"True positions: {true_positions}")
            
            # Get prediction from agent
            try:
                if args.agent_type == "MajorityVotingAgents":
                    pred = agent.process(swap_sequence, query)
                elif args.agent_type == "ChainOfAgents":
                    pred = agent.process(swap_sequence, query, extraction_func=extract_position_dict)
                elif args.agent_type == "PrefixSumAgents":
                    pred = agent.hierarchical_process(swap_sequence, query, extraction_func=extract_position_dict)
                
                logger.info(f"Raw prediction: {pred}")
                
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
                
                logger.info(f"Parsed prediction: {predicted_positions}")
                
                # Add debugging for dictionary composition failures
                if predicted_positions is None:
                    logger.warning(f"FAILED PARSING: Could not extract dictionary from: {pred[:100]}...")
                elif not isinstance(predicted_positions, dict):
                    logger.warning(f"FAILED PARSING: Extracted non-dict: {type(predicted_positions)} = {predicted_positions}")
                
                # Evaluate accuracy
                exact_match, element_accuracy = evaluate_permutation_accuracy(predicted_positions, true_positions)
                exact_match_results.append(int(exact_match))
                element_accuracy_results.append(element_accuracy)
                
                logger.info(f"Exact match: {exact_match}, Element accuracy: {element_accuracy:.3f}")
                
            except Exception as e:
                logger.error(f"Error processing run {run_idx+1}: {str(e)}")
                exact_match_results.append(0)
                element_accuracy_results.append(0.0)
        
        # Calculate and log averages for this element count
        avg_exact_match = mean(exact_match_results)
        avg_element_accuracy = mean(element_accuracy_results)
        max_exact_match = max(exact_match_results)
        max_element_accuracy = max(element_accuracy_results)
        
        logger.info(f"Swaps={num_swaps}, AvgExactMatch={avg_exact_match:.3f}, AvgElementAccuracy={avg_element_accuracy:.3f}")
        logger.info(f"Swaps={num_swaps}, MaxExactMatch={max_exact_match:.3f}, MaxElementAccuracy={max_element_accuracy:.3f}")
        
        wandb.log({
            "avg_exact_match": avg_exact_match,
            "avg_element_accuracy": avg_element_accuracy,
            "max_exact_match": max_exact_match,
            "max_element_accuracy": max_element_accuracy,
            "num_swaps": num_swaps
        })


if __name__ == "__main__":
    main()