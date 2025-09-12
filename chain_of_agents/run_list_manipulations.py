import argparse
import random
import wandb
from statistics import mean, stdev
import math

from logger import setup_logger
from main import MajorityVotingAgents, ChainOfAgents, ListManipulationPrefixSumAgents
from utils import *
import tabulate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_type", choices=["MajorityVotingAgents", "ChainOfAgents", "ListManipulationPrefixSumAgents"], default="ListManipulationPrefixSumAgents", help="Type of agent to use")
    parser.add_argument("--num_agents", type=int, default=3, help="Number of agents to use in MajVote setup")
    parser.add_argument("--model_type", type=str, default="meta-llama/Llama-3.3-70B-Instruct-Turbo", help="Model type to use for agents")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max tokens for each agent")
    parser.add_argument("--chunk_size", type=int, default=2, help="Chunk size for Chain of Agents (number of operations)")
    parser.add_argument("--num_runs", type=int, default=5, help="Number of runs to perform")
    parser.add_argument("--num_elements", type=int, default=5, help="Number of elements in list")
    parser.add_argument("--max_operation_types", type=int, default=2, help="Maximum number of operation types to use (1-7)")
    parser.add_argument("--min_sequence_length", type=int, default=2, help="Minimum sequence length for testing")
    parser.add_argument("--max_sequence_length", type=int, default=4, help="Maximum sequence length for testing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--branching_factor", type=int, default=2, help="Branching factor for prefix sum agents")
    parser.add_argument("--use_python_code", type=lambda x: x.lower() in ['true', '1', 'yes', 'on'], default=True, help="Use Python code strings instead of natural language")
    parser.add_argument("--enable_wandb", type=lambda x: x.lower() in ['true', '1', 'yes', 'on'], default=True, help="Enable Weights & Biases logging (true/false)")
    args = parser.parse_args()

    # Create a nice table showing all arguments
    args_table = []
    for arg, value in vars(args).items():
        args_table.append([arg, value])
        
    print("\n" + "="*50)
    print("LIST MANIPULATION EXPERIMENT CONFIGURATION")
    print("="*50)
    print(tabulate.tabulate(args_table, headers=["Argument", "Value"], tablefmt="grid"))
    print("="*50 + "\n")
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    name_dict = {
        "MajorityVotingAgents": "maj-voting",
        "ChainOfAgents": "coa",
        "ListManipulationPrefixSumAgents": "list-prefix-sum"
    }
    
    if args.agent_type == "MajorityVotingAgents":
        run_name = f"{name_dict[args.agent_type]}_elements{args.num_elements}_agents{args.num_agents}_seq{args.min_sequence_length}-{args.max_sequence_length}"
    elif args.agent_type == "ChainOfAgents":
        run_name = f"{name_dict[args.agent_type]}_elements{args.num_elements}_chunk{args.chunk_size}_seq{args.min_sequence_length}-{args.max_sequence_length}"
    elif args.agent_type == "ListManipulationPrefixSumAgents":
        run_name = f"{name_dict[args.agent_type]}_elements{args.num_elements}_b{args.branching_factor}_seq{args.min_sequence_length}-{args.max_sequence_length}"
    
    if args.enable_wandb:
        wandb.init(project="coa-list-manipulation-eval", config=vars(args), name=run_name, reinit=True)
    else:
        print("Weights & Biases logging disabled")
    logger = setup_logger(enable_wandb=args.enable_wandb)

    # Initialize agents
    if args.agent_type == "MajorityVotingAgents":
        # Simplified prompt for majority voting on list manipulation
        prompt = """You are a reasoning agent that tracks list transformations through Python operations.

Your task: Apply each Python operation sequentially to determine the final index mapping.

**PROCESS:**
1. Start with identity mapping: [a[0], a[1], a[2], ...] based on list size
2. Apply each operation step-by-step 
3. Show your work for each step
4. Give final answer in format: [a[i], a[j], a[k], ...]

**EXAMPLES:**

**Example 1:** Operations: a[0], a[1] = a[1], a[0] | a[::-1]
Step 1: Start with [a[0], a[1], a[2]] 
Step 2: a[0], a[1] = a[1], a[0] → [a[1], a[0], a[2]]
Step 3: a[::-1] → [a[2], a[0], a[1]]
The answer is: [a[2], a[0], a[1]]

**Example 2:** Operations: a[:] = a[1:] + a[:1]
Step 1: Start with [a[0], a[1], a[2]]
Step 2: a[:] = a[1:] + a[:1] → [a[1], a[2], a[0]]
The answer is: [a[1], a[2], a[0]]

**KEY OPERATIONS:**
- a[i], a[j] = a[j], a[i] → Swap positions i and j
- a[::-1] → Reverse entire list
- a[:] = a[k:] + a[:k] → Rotate left by k positions

**INSTRUCTIONS:**
- Operations are separated by "|" symbol
- Always start with identity mapping based on list size
- Apply operations left-to-right in sequence
- Format final answer exactly as: "The answer is: [a[i1], a[i2], ...]" """
        
        agent = MajorityVotingAgents(
            num_agents=args.num_agents,
            model=args.model_type,
            max_tokens=args.max_tokens,
            prompt=prompt,
        )
    elif args.agent_type == "ChainOfAgents":
        # For ChainOfAgents, we'll need to add list manipulation support
        # This is simplified - in practice you'd need specialized prompts
        worker_prompt, manager_prompt = get_list_manipulation_prompts(b=2)
        agent = ChainOfAgents(
            worker_model=args.model_type,
            manager_model=args.model_type,
            chunk_size=args.chunk_size,
            worker_prompt=worker_prompt,
            manager_prompt=manager_prompt,
            max_tokens_worker=args.max_tokens,
            use_index_hints=False,
        )
    elif args.agent_type == "ListManipulationPrefixSumAgents":
        worker_prompt, manager_prompt = get_list_manipulation_prompts(b=args.branching_factor)
        agent = ListManipulationPrefixSumAgents(
            worker_model=args.model_type,
            manager_model=args.model_type,
            max_tokens_worker=args.max_tokens,
            max_tokens_manager=args.max_tokens,
            worker_prompt=worker_prompt,
            manager_prompt=manager_prompt,
            branching_factor=args.branching_factor,
        )

    # Fixed parameters
    complexity_level = args.max_operation_types
    sequence_lengths = range(args.min_sequence_length, args.max_sequence_length + 1)
    logger.info(f"Testing sequence lengths from {args.min_sequence_length} to {args.max_sequence_length} with fixed complexity {complexity_level}")
    
    # Run experiments for different sequence lengths
    for num_ops in sequence_lengths:
        logger.info(f"Testing {num_ops} operations with complexity level {complexity_level}")
        exact_match_results = []
        element_accuracy_results = []
        token_stats = []
        error_type_counts = {}
        
        for run_idx in range(args.num_runs):
            # Generate list manipulation problem
            operations_sequence, true_list = generate_list_manipulation_problem(
                n=args.num_elements, 
                num_ops=num_ops, 
                max_operation_types=complexity_level,
                use_python_code=args.use_python_code
            )
            original_list = list(range(1, args.num_elements + 1))
            
            query = f"""You are given a list with {args.num_elements} elements: {original_list}. 

After applying the sequence of operations, determine the final index mapping.

An index mapping shows where each original element ended up. For example:
- If the final list is [2, 1, 3], the index mapping is [a[1], a[0], a[2]]
- This means: position 0 has original element 2 (index 1), position 1 has original element 1 (index 0), position 2 has original element 3 (index 2)

What is the final index mapping after all the given operations?"""
            
            logger.info(f"Run {run_idx+1}/{args.num_runs} - Elements: {args.num_elements}, Operations: {num_ops}")
            logger.info(f"Operations sequence: {operations_sequence}")
            logger.info(f"True list: {true_list}")
            logger.info(f"Original list: {original_list}")
                
            # Calculate and log ground truth index mapping
            true_mapping = []
            for value in true_list:
                # Find original index of this value (value - 1 since original_list is 1-indexed but positions are 0-indexed)
                original_index = value - 1
                true_mapping.append(original_index)
            true_mapping_str = f"[{', '.join(f'a[{idx}]' for idx in true_mapping)}]"
            logger.info(f"Ground truth index mapping: {true_mapping_str}")
                
            # Get prediction from agent
            try:
                if args.agent_type == "MajorityVotingAgents":
                    result = agent.process(operations_sequence, query, extraction_func=extract_index_mapping)
                elif args.agent_type == "ChainOfAgents":
                    # Need to modify ChainOfAgents to handle list manipulations
                    result = agent.process(operations_sequence, query, extraction_func=extract_index_mapping)
                elif args.agent_type == "ListManipulationPrefixSumAgents":
                    result = agent.hierarchical_process(operations_sequence, query, extraction_func=extract_index_mapping)
                    
                pred = result['content']
                token_stats.append(result['token_usage'])
                logger.info(f"Raw prediction: {pred}")
                logger.info(f"Ground truth mapping: {true_mapping_str}")
                
                # Parse prediction
                if isinstance(pred, str):
                    predicted_mapping = parse_index_mapping(pred)
                else:
                    predicted_mapping = pred
                
                logger.info(f"Parsed prediction mapping: {predicted_mapping}")
                
                # Add debugging for parsing failures
                if predicted_mapping is None:
                    logger.warning(f"FAILED PARSING: Could not extract index mapping from: {pred[:100]}...")
                
                # Validate intermediate steps
                validation_results = validate_intermediate_steps(predicted_mapping, args.num_elements)
                logger.info(f"Validation: {validation_results}")
                
                # Evaluate accuracy with enhanced error detection
                exact_match, element_accuracy, error_type = evaluate_list_manipulation_accuracy(
                    predicted_mapping, true_list, original_list
                )
                exact_match_results.append(int(exact_match))
                element_accuracy_results.append(element_accuracy)
                
                # Track error types for analysis
                error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1
                
                logger.info(f"Exact match: {exact_match}, Element accuracy: {element_accuracy:.3f}, Error type: {error_type}")
                
            except Exception as e:
                logger.error(f"Error processing run {run_idx+1}: {str(e)}")
                exact_match_results.append(0)
                element_accuracy_results.append(0.0)
                error_type_counts["EXCEPTION"] = error_type_counts.get("EXCEPTION", 0) + 1
                # Add dummy token stats for failed runs
                token_stats.append({
                    'avg_completion_tokens': 0,
                    'max_completion_tokens': 0,
                    'avg_prompt_tokens': 0,
                    'max_prompt_tokens': 0
                })
            
        # Calculate statistics for this sequence length
        avg_exact_match = mean(exact_match_results)
        avg_element_accuracy = mean(element_accuracy_results)
            
        # Calculate standard error (SE = std_dev / sqrt(n))
        n_runs = len(exact_match_results)
        if n_runs > 1:
            std_exact_match = stdev(exact_match_results)
            std_element_accuracy = stdev(element_accuracy_results)
            se_exact_match = std_exact_match / math.sqrt(n_runs)
            se_element_accuracy = std_element_accuracy / math.sqrt(n_runs)
        else:
            se_exact_match = 0.0
            se_element_accuracy = 0.0
        
        # Log simplified metrics
        logger.info(f"Operations={num_ops}, ExactMatch={avg_exact_match:.3f}±{se_exact_match:.3f}, ElementAccuracy={avg_element_accuracy:.3f}±{se_element_accuracy:.3f}")
            
        # Log to wandb with simplified metrics
        if args.enable_wandb:
            wandb_data = {
                "avg_exact_match": avg_exact_match,
                "avg_element_accuracy": avg_element_accuracy,
                "se_exact_match": se_exact_match,
                "se_element_accuracy": se_element_accuracy,
                "num_ops": num_ops
            }
            wandb.log(wandb_data)


if __name__ == "__main__":
    main()