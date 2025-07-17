# main.py
import argparse
import random
import logging
import wandb
from statistics import mean

from logger import setup_logger
from main import MajorityVotingAgents, ChainOfAgents, PrefixSumAgents  # Assume these are defined elsewhere
from utils import split_into_chunks, get_default_prompts, get_majority_vote_prompt, extract_answer, get_prefix_sum_prompt, get_parity_prompt


def generate_bitstring(length):
    return ''.join(random.choice('01') for _ in range(length))

def compute_parity(bitstring):
    return str(bitstring.count('1') % 2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_type", choices=["MajorityVotingAgents", "ChainOfAgents", "PrefixSumAgents"], default="MajorityVotingAgents")
    parser.add_argument("--num_agents", type=int, default=5)
    parser.add_argument("--model_type", type=str, default="lgai/exaone-3-5-32b-instruct")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--chunk_size", type=int, default=10)
    parser.add_argument("--num_runs", type=int, default=5)
    parser.add_argument("--min_seq_length", type=int, default=16)
    parser.add_argument("--max_seq_length", type=int, default=64)
    parser.add_argument("--step", type=int, default=8)
    args = parser.parse_args()
    
    name_dict = {
        "MajorityVotingAgents": "maj-voting",
        "ChainOfAgents": "coa",
        "PrefixSumAgents": "prefix-sum"
    }
    run_name = f"{name_dict[args.agent_type]}_seq{args.min_seq_length}-{args.max_seq_length}_agents{args.num_agents}"
    wandb.init(project="coa-parity-eval", config=vars(args), name=run_name, reinit=True)
    logger = setup_logger()

    if args.agent_type == "MajorityVotingAgents":
        agent = MajorityVotingAgents(
            num_agents=args.num_agents,
            model=args.model_type,
            max_tokens=args.max_tokens
        )
    elif args.agent_type == "ChainOfAgents":
        worker_prompt, manager_prompt = get_parity_prompt()
        agent = ChainOfAgents(
            worker_model=args.model_type,
            manager_model=args.model_type,
            chunk_size=args.chunk_size,
            worker_prompt=worker_prompt,
            manager_prompt=manager_prompt,
            max_tokens_worker=args.max_tokens
        )
    elif args.agent_type == "PrefixSumAgents":
        agent = PrefixSumAgents(
            worker_model=args.model_type,
            manager_model=args.model_type,
            max_tokens_worker=args.max_tokens,
            max_tokens_manager=args.max_tokens
        )


    for seq_len in range(args.min_seq_length, args.max_seq_length + 1, args.step):
        results = []
        for _ in range(args.num_runs):
            bitstring = generate_bitstring(seq_len)
            query = "What is the parity of the given binary string?"
            input_text = ' '.join(bitstring)  # Convert bitstring to space-separated

            if args.agent_type == "MajorityVotingAgents":
                pred = agent.process(input_text, query)
            elif args.agent_type == "ChainOfAgents":
                pred = agent.process(input_text, query, extraction_func=extract_answer)
            elif args.agent_type == "PrefixSumAgents":
                pred = agent.hierarchical_process(input_text, query, extraction_func=extract_answer)
            pred = pred.strip().lower()
            pred = "0" if pred == "even" else "1" if pred == "odd" else pred
            print(f"Bitstring: {bitstring}, Predicted Parity: {pred}")
            truth = compute_parity(bitstring)
            print(f"Truth Parity: {truth}")
            correct = int(pred == truth)
            results.append(correct)
        avg_accuracy = mean(results)
        logger.info(f"SeqLen={seq_len}, AvgAccuracy={avg_accuracy:.3f}")
        wandb.log({"avg_accuracy": avg_accuracy, "sequence_length": seq_len})

if __name__ == "__main__":
    main()
