from typing import Optional, Iterator, Dict
from agents import WorkerAgent, ManagerAgent
from utils import *
import logging
import json
import random
import wandb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChainOfAgents:
    """Main class for the Chain of Agents implementation."""
    
    def __init__(
        self,
        worker_model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",  # Together AI model
        manager_model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",  # Together AI model
        chunk_size: int = 500,
        max_tokens_worker: int = 512,
        max_tokens_manager: int = 1024,
        worker_prompt: Optional[str] = None,
        manager_prompt: Optional[str] = None,
        use_index_hints: bool = False
    ):
        """
        Initialize the Chain of Agents.
        
        Args:
            worker_model: Model to use for worker agents
            manager_model: Model to use for manager agent
            chunk_size: Maximum tokens per chunk
            worker_prompt: Custom system prompt for workers
            manager_prompt: Custom system prompt for manager
        """
        default_worker_prompt, default_manager_prompt = get_default_prompts()
        
        self.worker_prompt = worker_prompt or default_worker_prompt
        self.manager_prompt = manager_prompt or default_manager_prompt
        self.chunk_size = chunk_size
        self.worker_model = worker_model
        self.manager_model = manager_model
        self.max_tokens_worker = max_tokens_worker
        self.max_tokens_manager = max_tokens_manager
        self.use_index_hints = use_index_hints

        logger.info(f"Initialized Chain of Agents with {worker_model} workers and {manager_model} manager")
    
    def _split_swaps_into_chunks(self, swap_sequence: str, chunk_size: int) -> List[str]:
        """
        Split swap sequence into chunks by number of swaps.
        
        Args:
            swap_sequence: String with swap commands separated by " | "
            chunk_size: Number of swaps per chunk
            
        Returns:
            List[str]: List of chunks, each containing up to chunk_size swaps
        """
        swaps = swap_sequence.split(" | ")
        chunks = []
        
        for i in range(0, len(swaps), chunk_size):
            chunk_swaps = swaps[i:i + chunk_size]
            chunk = " | ".join(chunk_swaps)
            chunks.append(chunk)
        
        return chunks
    
    def process(self, input_text: str, query: str, extraction_func = None ) -> str:
        """
        Process a long text input using the Chain of Agents.
        
        Args:
            input_text: The long input text to process
            query: The user's query about the text
            
        Returns:
            str: The final response from the manager agent
        """
        # Split text into chunks
        if self.use_index_hints:
            chunks = split_binary_string(input_text, self.chunk_size)
        elif "Swap ball" in input_text:  # Permutation problem
            chunks = self._split_swaps_into_chunks(input_text, self.chunk_size)
        else:
            chunks = split_into_chunks(input_text, self.chunk_size)
        # Process chunks with worker agents
        worker_outputs = []
        worker_token_usages = []
        worker_prompt_tokens = []
        previous_cu = None
        current_positions = None
        
        # Initialize positions for permutation problems
        if "Swap ball" in input_text:
            # Extract number of balls from the input
            import re
            ball_numbers = re.findall(r'ball (\d+)', input_text)
            max_ball = max(int(num) for num in ball_numbers) if ball_numbers else 5
            current_positions = {i: i for i in range(1, max_ball + 1)}
        
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            logger.info(f"Chunk content: {chunk}")  # Debug log
            worker = WorkerAgent(self.worker_model, self.worker_prompt, max_tokens=self.max_tokens_worker)
            
            # For permutation problems, include current state
            if "Swap ball" in input_text and current_positions:
                chunk_with_state = f"Current positions: {current_positions}\nSwap operations: {chunk}"
                response = worker.process_chunk(chunk_with_state, query)
            else:
                response = worker.process_chunk(chunk, query)
                
            if extraction_func:
                output = extraction_func(response["content"])
                # Update current positions for next chunk
                if "Swap ball" in input_text and output:
                    from utils import parse_position_dict
                    parsed_positions = parse_position_dict(output)
                    if parsed_positions:
                        current_positions = parsed_positions
            worker_outputs.append(output)

            worker_token_usages.append(response["usage"].completion_tokens)
            chunk_prompt_tokens = getattr(response["usage"], 'prompt_tokens', 0)
            worker_prompt_tokens.append(chunk_prompt_tokens)
            logger.info(f"Worker chunk {i+1} prompt tokens: {chunk_prompt_tokens}")
            if wandb.run:
                wandb.log({f"worker_chunk_{i+1}_prompt_tokens": chunk_prompt_tokens})

        if worker_token_usages:
            avg_worker_tokens = sum(worker_token_usages) / len(worker_token_usages)
            max_worker_tokens = max(worker_token_usages)
            logger.info(f"Average worker completion tokens: {avg_worker_tokens:.2f}")
            logger.info(f"Max worker completion tokens: {max_worker_tokens}")
            
            avg_worker_prompt_tokens = sum(worker_prompt_tokens) / len(worker_prompt_tokens)
            max_worker_prompt_tokens = max(worker_prompt_tokens)
            logger.info(f"Average worker prompt tokens: {avg_worker_prompt_tokens:.2f}")
            logger.info(f"Max worker prompt tokens: {max_worker_prompt_tokens}")
            
            if wandb.run:
                wandb.log({
                    "average_worker_completion_tokens": avg_worker_tokens, 
                    "max_worker_completion_tokens": max_worker_tokens,
                    "average_worker_prompt_tokens": avg_worker_prompt_tokens,
                    "max_worker_prompt_tokens": max_worker_prompt_tokens
                })

        # Synthesize results with manager agent
        manager = ManagerAgent(self.manager_model, self.manager_prompt, max_tokens=self.max_tokens_manager)
        manager_response = manager.synthesize(worker_outputs, query)
        if extraction_func:
            final_output = extraction_func(manager_response["content"])

        manager_completion_tokens = manager_response["usage"].completion_tokens
        manager_prompt_tokens = getattr(manager_response["usage"], 'prompt_tokens', 0)
        logger.info(f"Manager completion tokens: {manager_completion_tokens}")
        logger.info(f"Manager prompt tokens: {manager_prompt_tokens}")
        if wandb.run:
            wandb.log({"manager_completion_tokens": manager_completion_tokens, "manager_prompt_tokens": manager_prompt_tokens})
        return final_output 
    
    def process_stream(self, input_text: str, query: str) -> Iterator[Dict[str, str]]:
        """Process text with streaming - yields worker and manager messages."""
        worker_outputs = []
        previous_cu = None
        
        chunks = split_into_chunks(input_text, self.chunk_size, self.worker_model)
        total_chunks = len(chunks)
        
        # Debug logging for metadata
        metadata_message = {
            "type": "metadata",
            "content": json.dumps({
                "total_chunks": total_chunks,
                "total_pages": getattr(input_text, 'total_pages', 0)
            })
        }
        logger.info(f"Sending metadata: {metadata_message}")  # Debug log
        yield metadata_message
        
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{total_chunks}")
            worker = WorkerAgent(self.worker_model, self.worker_prompt)
            output = worker.process_chunk(chunk, query, previous_cu)
            worker_outputs.append(output)
            previous_cu = output
            
            # Debug logging for worker message
            worker_message = {
                "type": "worker",
                "content": output,
                "progress": {
                    "current": i + 1,
                    "total": total_chunks
                }
            }
            logger.info(f"Sending worker message: {worker_message}")  # Debug log
            yield worker_message
        
        logger.info("Processing manager synthesis")
        manager = ManagerAgent(self.manager_model, self.manager_prompt)
        final_output = manager.synthesize(worker_outputs, query)
        
        yield {
            "type": "manager",
            "content": final_output
        } 


class MajorityVotingAgents:
    """Class for implementing majority voting among multiple agent instances."""
    
    def __init__(
        self,
        model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",  # Together AI model
        num_agents: int = 3,
        max_tokens: int = 512,
        prompt: Optional[str] = None,
    ):
        """
        Initialize the Majority Voting Agents.
        
        Args:
            model: Model to use for all agents
            num_agents: Number of agents to create for voting
            prompt: Custom system prompt for agents
        """
        default_prompt =  get_majority_vote_prompt()
        self.prompt = prompt or default_prompt
        self.model = model
        self.num_agents = num_agents
        self.max_tokens = max_tokens 
        
        logger.info(f"Initialized Majority Voting with {self.num_agents} agents using model {self.model}")
    
    def process(self, input_text: str, query: str) -> str:
        agent_answers = []
        token_usages = []
        prompt_token_usages = []

        for agent_num in range(self.num_agents):
            logger.info(f"Running agent {agent_num+1}/{self.num_agents}")
            
            worker = WorkerAgent(self.model, self.prompt, max_tokens=self.max_tokens)
            result = worker.process_chunk(input_text, query, None)
            
            answer = extract_answer(result["content"])
            if answer:
                agent_answers.append(answer)
            else:
                logger.warning(f"Agent {agent_num+1} returned no valid answer")
            
            completion_tokens = result["usage"].completion_tokens
            prompt_tokens = getattr(result["usage"], 'prompt_tokens', 0)
            token_usages.append(completion_tokens)
            prompt_token_usages.append(prompt_tokens)
            if wandb.run:
                wandb.log({f"agent_{agent_num+1}_completion_tokens": completion_tokens})
            
            del worker

        # Log token stats
        if token_usages:
            avg_tokens = sum(token_usages) / len(token_usages)
            max_tokens = max(token_usages)
            logger.info(f"Average completion tokens: {avg_tokens:.2f}")
            logger.info(f"Max completion tokens: {max_tokens}")
            
            avg_prompt_tokens = sum(prompt_token_usages) / len(prompt_token_usages)
            max_prompt_tokens = max(prompt_token_usages)
            logger.info(f"Average prompt tokens: {avg_prompt_tokens:.2f}")
            logger.info(f"Max prompt tokens: {max_prompt_tokens}")
            
            if wandb.run:
                wandb.log({
                    "average_completion_tokens": avg_tokens, 
                    "max_completion_tokens": max_tokens,
                    "average_prompt_tokens": avg_prompt_tokens,
                    "max_prompt_tokens": max_prompt_tokens
                })

        answers = {}
        for output in agent_answers:
            answers[output] = answers.get(output, 0) + 1

        most_common_answer = max(answers.items(), key=lambda x: x[1])[0]
        
        return most_common_answer



class PrefixSumAgents:
    """Class for implementing prefix sum calculation using multiple agents."""
    
    def __init__(
            self,
            worker_model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            manager_model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            max_tokens_worker: int = 512,
            max_tokens_manager: int = 1024,
            worker_prompt: Optional[str] = None,
            manager_prompt: Optional[str] = None,
            branching_factor: int = 2  # Branching factor for hierarchical processing 
            ):
        default_worker_prompt, default_manager_prompt = get_prefix_sum_prompt(b=branching_factor)
        self.worker_model = worker_model
        self.manager_model = manager_model 
        self.max_tokens_worker = max_tokens_worker
        self.max_tokens_manager = max_tokens_manager
        self.worker_prompt = worker_prompt or default_worker_prompt
        self.manager_prompt = manager_prompt or default_manager_prompt
        self.b = branching_factor  # Branching factor for hierarchical processing

    def hierarchical_process(self, input_text: str, query: str, extraction_func=None) -> str:
        """
        Hierarchical processing of input using b-ary tree agent structure.

        Args:
            input_text: Binary string of 0s and 1s
            query: Query for each agent
            extraction_func: Optional function to extract/transform output at each step
            b: Branching factor (number of children each manager processes)
        
        Returns:
            str: Final synthesized output
        """
        logging.info(f"Starting hierarchical_process with input: {input_text} and query: '{query}', branching factor: {self.b}")

        #level_outputs = []
        #for i, token in enumerate(input_text.replace(" ", "")):
        #    worker = WorkerAgent(self.worker_model, self.worker_prompt, max_tokens=self.max_tokens_worker)
        #    output = worker.process_chunk(token, query, None)
        #    logging.info(f"Worker {i} output: {output}")
        #    if extraction_func:
        #        output = extraction_func(output)
        #        logging.info(f"Worker {i} extracted output: {output}")
        #    level_outputs.append(output)
        #    del worker

        # Handle different input types
        if "Swap ball" in input_text:  # Permutation problem
            swaps = input_text.split(" | ")  # Split by swap separators
            # Initialize positions for permutation
            import re
            ball_numbers = re.findall(r'ball (\d+)', input_text)
            max_ball = max(int(num) for num in ball_numbers) if ball_numbers else 5
            initial_positions = {i: i for i in range(1, max_ball + 1)}
            
            # Process each swap individually with WorkerAgent to get actual results
            level_outputs = []
            current_positions = initial_positions.copy()
            for swap in swaps:
                worker = WorkerAgent(self.worker_model, self.worker_prompt, max_tokens=self.max_tokens_worker)
                # Format input for worker: current positions + single swap
                worker_input = f"Current positions: {current_positions}\nSwap operation: {swap}"
                response = worker.process_chunk(worker_input, query)
                
                if extraction_func:
                    extracted = extraction_func(response["content"])
                    # Update current_positions for next swap
                    from utils import parse_position_dict
                    parsed = parse_position_dict(extracted)
                    if parsed:
                        current_positions = parsed
                    level_outputs.append(extracted)
                else:
                    level_outputs.append(response["content"])
                del worker
        else:  # Binary string (original functionality)
            level_outputs = [digit for digit in input_text.replace(" ", "")]
        logging.info(f"Initial level outputs: {level_outputs}")
    
        round_num = 0
        sum_of_max_completion_tokens = 0
        sum_of_avg_completion_tokens = 0
        sum_of_max_prompt_tokens = 0
        sum_of_avg_prompt_tokens = 0
        is_permutation = "Swap ball" in input_text
        
        while len(level_outputs) > 1:
            logging.info(f"Manager round {round_num} with {len(level_outputs)} inputs")
            next_level = []
            completion_token_usages = []
            prompt_token_usages = []
            
            for i in range(0, len(level_outputs), self.b):
                manager = ManagerAgent(self.manager_model, self.manager_prompt, max_tokens=self.max_tokens_manager)
                chunk = level_outputs[i:i+self.b]
                
                # Process chunk with manager agent (same for both permutation and binary)
                output = manager.synthesize(chunk, query)
                logging.info(f"Manager input: {chunk} -> {output['content']}")
                
                if extraction_func:
                    synthesized = extraction_func(output["content"])
                else:
                    synthesized = output["content"]
                    
                next_level.append(synthesized)
                
                completion_tokens = output["usage"].completion_tokens
                prompt_tokens = getattr(output["usage"], 'prompt_tokens', 0)
                completion_token_usages.append(completion_tokens)
                prompt_token_usages.append(prompt_tokens)
                
                del manager
                
            # Accumulate token usage stats for this round
            if completion_token_usages:
                avg_completion_tokens = sum(completion_token_usages) / len(completion_token_usages)
                max_completion_tokens = max(completion_token_usages)
                sum_of_avg_completion_tokens += avg_completion_tokens
                sum_of_max_completion_tokens += max_completion_tokens
                
            if prompt_token_usages:
                avg_prompt_tokens = sum(prompt_token_usages) / len(prompt_token_usages)
                max_prompt_tokens = max(prompt_token_usages)
                sum_of_avg_prompt_tokens += avg_prompt_tokens
                sum_of_max_prompt_tokens += max_prompt_tokens
                
            level_outputs = next_level
            round_num += 1
            
        # Log accumulated token statistics after while loop terminates
        logger.info(f"Sum of max completion tokens across all levels: {sum_of_max_completion_tokens}")
        logger.info(f"Sum of average completion tokens across all levels: {sum_of_avg_completion_tokens:.2f}")
        logger.info(f"Sum of max prompt tokens across all levels: {sum_of_max_prompt_tokens}")
        logger.info(f"Sum of average prompt tokens across all levels: {sum_of_avg_prompt_tokens:.2f}")
        
        if wandb.run:
            wandb.log({
                "sum_of_max_completion_tokens": sum_of_max_completion_tokens,
                "sum_of_avg_completion_tokens": sum_of_avg_completion_tokens,
                "sum_of_max_prompt_tokens": sum_of_max_prompt_tokens,
                "sum_of_avg_prompt_tokens": sum_of_avg_prompt_tokens
            })

        logging.info(f"Final output: {level_outputs[0]}")
        return level_outputs[0]


def test_majority_voting(query, input_text):
    maj_vote = MajorityVotingAgents(num_agents=5, max_tokens=2048)
    result = maj_vote.process(input_text, query)
    print(f"Majority Voting Result: {result}")

def test_coa(query, input_text):
    parity_worker_prompt, parity_manager_prompt = get_parity_prompt()
    coa = ChainOfAgents(
        worker_model="lgai/exaone-3-5-32b-instruct",
        manager_model="lgai/exaone-3-5-32b-instruct",
        chunk_size=10,
        worker_prompt=parity_worker_prompt,
        manager_prompt=parity_manager_prompt,
        max_tokens_worker=1024
    )
    final_output = coa.process(input_text, query, extraction_func=extract_answer)
    print(f"Chain-of-Agents Result: {final_output}")

def test_prefix_sum(query, input_text):
    prefix_sum_agents = PrefixSumAgents(
        worker_model="lgai/exaone-3-5-32b-instruct",
        manager_model="lgai/exaone-3-5-32b-instruct",
        max_tokens_worker=64,
        max_tokens_manager=64
    )
    final_output = prefix_sum_agents.hierarchical_process(input_text, query, extraction_func=extract_answer)
    print(f"Prefix Sum Result: {final_output}")

if __name__ == "__main__":
    seq_len = 32 
    input_text = ' '.join(random.choice(['0', '1']) for _ in range(seq_len))
    query = "What is the parity of the given binary string?"

    #test_majority_voting(query, input_text)
    #test_coa(query, input_text)
    test_prefix_sum(query, input_text)

    parity = "even" if input_text.count('1') % 2 == 0 else "odd"
    print(f"Ground truth parity: {parity}")
    print(f"Number of 1s in input: {input_text.count('1')}")
