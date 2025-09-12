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
    parser.add_argument("--agent_type", choices=["MajorityVotingAgents", "ChainOfAgents", "ListManipulationPrefixSumAgents"], default="MajorityVotingAgents", help="Type of agent to use")
    parser.add_argument("--num_agents", type=int, default=3, help="Number of agents to use in MajVote setup")
    parser.add_argument("--model_type", type=str, default="lgai/exaone-3-5-32b-instruct", help="Model type to use for agents")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max tokens for each agent")
    parser.add_argument("--chunk_size", type=int, default=2, help="Chunk size for Chain of Agents (number of operations)")
    parser.add_argument("--num_runs", type=int, default=5, help="Number of runs to perform")
    parser.add_argument("--num_elements", type=int, default=4, help="Number of elements in list")
    parser.add_argument("--min_ops", type=int, default=4, help="Minimum number of operations")
    parser.add_argument("--max_ops", type=int, default=4, help="Maximum number of operations")
    parser.add_argument("--max_operation_types", type=int, default=1, help="Maximum number of operation types to use (1-7)")
    parser.add_argument("--progressive_complexity", action="store_true", help="Enable progressive complexity testing")
    parser.add_argument("--start_complexity", type=int, default=1, help="Starting complexity level (1-7)")
    parser.add_argument("--end_complexity", type=int, default=1, help="Ending complexity level (1-7)")
    parser.add_argument("--test_sequence_lengths", action="store_true", help="Test different sequence lengths")
    parser.add_argument("--min_sequence_length", type=int, default=1, help="Minimum sequence length for testing")
    parser.add_argument("--max_sequence_length", type=int, default=10, help="Maximum sequence length for testing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--branching_factor", type=int, default=2, help="Branching factor for prefix sum agents")
    parser.add_argument("--use_python_code", type=bool, default=True, help="Use Python code strings instead of natural language")
    parser.add_argument("--enable_wandb", type=lambda x: x.lower() in ['true', '1', 'yes', 'on'], default=False, help="Enable Weights & Biases logging (true/false)")
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
        run_name = f"{name_dict[args.agent_type]}_elements{args.num_elements}_agents{args.num_agents}_ops{args.min_ops}-{args.max_ops}"
    elif args.agent_type == "ChainOfAgents":
        run_name = f"{name_dict[args.agent_type]}_elements{args.num_elements}_chunk{args.chunk_size}_ops{args.min_ops}-{args.max_ops}"
    elif args.agent_type == "ListManipulationPrefixSumAgents":
        run_name = f"{name_dict[args.agent_type]}_elements{args.num_elements}_b{args.branching_factor}_ops{args.min_ops}-{args.max_ops}"
    
    if args.enable_wandb:
        wandb.init(project="coa-list-manipulation-eval", config=vars(args), name=run_name, reinit=True)
    else:
        print("Weights & Biases logging disabled")
    logger = setup_logger(enable_wandb=args.enable_wandb)

    # Initialize agents
    if args.agent_type == "MajorityVotingAgents":
        # Create a prompt for majority voting on list manipulation with few-shot examples
        prompt = """You are a reasoning agent responsible for tracking list transformations through a sequence of Python operations with step-by-step verification.

Your task is to determine the final index mapping after performing all the given Python operations on a list.

**CHAIN-OF-THOUGHT PROCESS:**
1. Parse each Python operation carefully
2. Apply operations sequentially with intermediate verification
3. Double-check your final result
4. Output in the exact required format

**COMPREHENSIVE EXAMPLES:**

**Example 1 (Complex sequence):**
Input: a[0], a[2] = a[2], a[0] | a[::-1] | a[:] = a[1:] + a[:1]
Step 1: Start with a = [a[0], a[1], a[2], a[3], a[4]] (identity mapping)
Step 2: a[0], a[2] = a[2], a[0] → a = [a[2], a[1], a[0], a[3], a[4]]
Step 3: a[::-1] → a = [a[4], a[3], a[0], a[1], a[2]]
Step 4: a[:] = a[1:] + a[:1] → a = [a[3], a[0], a[1], a[2], a[4]]
Verification: Applied 3 operations sequentially, each step follows Python semantics
The answer is: [a[3], a[0], a[1], a[2], a[4]]

**Example 2 (With slicing):**
Input: a[:] = a[-2:] + a[:-2] | a[1], a[3] = a[3], a[1]
Step 1: Start with a = [a[0], a[1], a[2], a[3], a[4]] (identity mapping)
Step 2: a[:] = a[-2:] + a[:-2] → a = [a[3], a[4], a[0], a[1], a[2]]
Step 3: a[1], a[3] = a[3], a[1] → a = [a[3], a[1], a[0], a[4], a[2]]
Verification: Right rotation by 2, then swap positions 1 and 3
The answer is: [a[3], a[1], a[0], a[4], a[2]]

**Example 3 (Subrange operations):**
Input: a[1:4] = reversed(a[1:4]) | a[::2], a[1::2] = a[1::2], a[::2]
Step 1: Start with a = [a[0], a[1], a[2], a[3], a[4]] (identity mapping)
Step 2: a[1:4] = reversed(a[1:4]) → a = [a[0], a[3], a[2], a[1], a[4]]
Step 3: a[::2], a[1::2] = a[1::2], a[::2] → Even/odd swap → a = [a[3], a[0], a[2], a[1], a[4]]
Verification: Reverse middle elements, then swap even/odd positioned elements
The answer is: [a[3], a[0], a[2], a[1], a[4]]

**ERROR RECOVERY INSTRUCTIONS:**
- If you cannot parse a Python operation, output "PARSE_ERROR: [operation description]" and continue with remaining operations
- If indices are out of bounds, output "INDEX_ERROR: [operation description]" and skip that operation
- Always verify your final mapping has the correct number of elements
- Double-check that all indices are valid (0 to n-1 for n elements)

**PYTHON OPERATION REFERENCE:**
- a[i], a[j] = a[j], a[i] → Swap elements at positions i and j
- a[::-1] → Reverse entire list
- a[:] = a[k:] + a[:k] → Rotate left by k positions  
- a[:] = a[-k:] + a[:-k] → Rotate right by k positions
- a[i:j] = reversed(a[i:j]) → Reverse subrange from i to j-1
- a[:] = a[::2] + a[1::2] → Even indices first, then odd indices
- a = [a[i] for i in indices] → Reorder using index pattern

**CRITICAL VALIDATION STEPS:**
- Count elements: Final mapping must have same length as input
- Check indices: All indices must be valid (0 to n-1)
- Verify format: Must be exactly "[a[i1], a[i2], a[i3], ...]"
- Test logic: Each operation should produce a reasonable transformation

**INSTRUCTIONS:**
- Always start with identity mapping a = [a[0], a[1], a[2], ...] based on list size
- Process each Python operation in sequence, showing intermediate states
- Include verification step before final answer
- Use step-by-step template format exactly
- Handle errors gracefully with specific error messages
- Present the final answer in the format "The answer is: [a[i1], a[i2], a[i3], ...]" where i1, i2, i3 are the original indices
- Be consistent across multiple attempts - same input should produce same output"""
        
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

    # Determine test parameters based on settings
    if args.progressive_complexity:
        complexity_levels = range(args.start_complexity, args.end_complexity + 1)
        logger.info(f"Running progressive complexity testing from {args.start_complexity} to {args.end_complexity}")
    else:
        complexity_levels = [args.max_operation_types]
    
    if args.test_sequence_lengths:
        sequence_lengths = range(args.min_sequence_length, args.max_sequence_length + 1)
        logger.info(f"Testing sequence lengths from {args.min_sequence_length} to {args.max_sequence_length}")
    else:
        sequence_lengths = range(args.min_ops, args.max_ops + 1)
    
    # Run experiments for different complexities and sequence lengths
    for complexity_level in complexity_levels:
        for num_ops in sequence_lengths:
            logger.info(f"Testing complexity level {complexity_level} with {num_ops} operations")
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
                
                query = f"You are given a {args.num_elements} sized list. What is the final index mapping after all the given operations?"
                
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
            
            # Calculate statistics for this complexity/operation combination
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
            
            # Analyze error patterns
            error_analysis = analyze_error_patterns(error_type_counts, n_runs)
            
            # Log error type distribution and analysis
            logger.info(f"Complexity={complexity_level}, Operations={num_ops}, Error types: {error_type_counts}")
            logger.info(f"Complexity={complexity_level}, Operations={num_ops}, Success rate: {error_analysis['success_rate']:.3f}")
            if error_analysis['most_common_error']:
                logger.info(f"Complexity={complexity_level}, Operations={num_ops}, Most common error: {error_analysis['most_common_error'][0]} ({error_analysis['most_common_error'][1]} occurrences)")
            
            logger.info(f"Complexity={complexity_level}, Operations={num_ops}, AvgExactMatch={avg_exact_match:.3f}±{se_exact_match:.3f}, AvgElementAccuracy={avg_element_accuracy:.3f}±{se_element_accuracy:.3f}")
            logger.info(f"Complexity={complexity_level}, Operations={num_ops}, MaxExactMatch={max_exact_match:.3f}, MaxElementAccuracy={max_element_accuracy:.3f}")
            
            # Calculate average token statistics for this complexity/operation combination
            if token_stats:
                avg_completion_tokens = mean([stats['avg_completion_tokens'] for stats in token_stats])
                max_completion_tokens = max([stats['max_completion_tokens'] for stats in token_stats])
                avg_prompt_tokens = mean([stats['avg_prompt_tokens'] for stats in token_stats])
                max_prompt_tokens = max([stats['max_prompt_tokens'] for stats in token_stats])
                
                logger.info(f"Complexity={complexity_level}, Operations={num_ops}, AvgCompletionTokens={avg_completion_tokens:.2f}")
                logger.info(f"Complexity={complexity_level}, Operations={num_ops}, MaxCompletionTokens={max_completion_tokens:.2f}")
                logger.info(f"Complexity={complexity_level}, Operations={num_ops}, AvgPromptTokens={avg_prompt_tokens:.2f}")
                logger.info(f"Complexity={complexity_level}, Operations={num_ops}, MaxPromptTokens={max_prompt_tokens:.2f}")
                
                if args.enable_wandb:
                    wandb_data = {
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
                        "num_ops": num_ops,
                        "complexity_level": complexity_level
                    }
                    # Add error type counts to wandb
                    for error_type, count in error_type_counts.items():
                        wandb_data[f"error_count_{error_type.lower()}"] = count
                    wandb.log(wandb_data)
            else:
                if args.enable_wandb:
                    wandb_data = {
                        "avg_exact_match": avg_exact_match,
                        "avg_element_accuracy": avg_element_accuracy,
                        "max_exact_match": max_exact_match,
                        "max_element_accuracy": max_element_accuracy,
                        "std_exact_match": std_exact_match,
                        "std_element_accuracy": std_element_accuracy,
                        "se_exact_match": se_exact_match,
                        "se_element_accuracy": se_element_accuracy,
                        "num_ops": num_ops,
                        "complexity_level": complexity_level
                    }
                    # Add error type counts to wandb
                    for error_type, count in error_type_counts.items():
                        wandb_data[f"error_count_{error_type.lower()}"] = count
                    wandb.log(wandb_data)


if __name__ == "__main__":
    main()