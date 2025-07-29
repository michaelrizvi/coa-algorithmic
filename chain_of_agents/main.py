from typing import Optional, Iterator, Dict
from agents import WorkerAgent, ManagerAgent
from utils import split_into_chunks, get_default_prompts, get_majority_vote_prompt, extract_answer, get_prefix_sum_prompt, get_parity_prompt
import logging
import json
import random

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
        manager_prompt: Optional[str] = None
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
        
        logger.info(f"Initialized Chain of Agents with {worker_model} workers and {manager_model} manager")
    
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
        chunks = split_into_chunks(input_text, self.chunk_size, self.worker_model)
        
        # Process chunks with worker agents
        worker_outputs = []
        previous_cu = None
        
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            worker = WorkerAgent(self.worker_model, self.worker_prompt, max_tokens=self.max_tokens_worker)
            output = worker.process_chunk(chunk, query, previous_cu)
            print("Agent output:", output)  # Debug log
            if extraction_func:
                output = extraction_func(output)
            worker_outputs.append(output)
            previous_cu = output
        
        # Synthesize results with manager agent
        manager = ManagerAgent(self.manager_model, self.manager_prompt, max_tokens=self.max_tokens_manager)
        final_output = manager.synthesize(worker_outputs, query)
        
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
        """
        Process input using multiple agents and perform majority voting.
        
        Args:
            input_text: The input text to process
            query: The user's query about the text
            
        Returns:
            str: The most common response from the agents
        """
        # Create multiple agent instances and collect their outputs
        agent_answers = []
        
        for agent_num in range(self.num_agents):
            logger.info(f"Running agent {agent_num+1}/{self.num_agents}")
            
            # Create worker agent and process the complete input
            worker = WorkerAgent(self.model, self.prompt, max_tokens=self.max_tokens)
            result = worker.process_chunk(input_text, query, None)
            #print("Agent output:", result)  # Debug log
            answer = extract_answer(result)
            if answer:
                agent_answers.append(answer)
            else:
                logger.warning(f"Agent {agent_num+1} returned no valid answer")
            del worker  # Clean up worker instance
            
        # Perform majority voting on results
        answers = {}
        for output in agent_answers:
            if output in answers:
                answers[output] += 1
            else:
                answers[output] = 1
        
        # Find the most common answer
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
            manager_prompt: Optional[str] = None 
            ):
        """
        Initialize the Prefix Sum Agents.
        
        Args:
            model: Model to use for all agents
            num_agents: Number of agents to create for prefix sum calculation
        """
        default_worker_prompt, default_manager_prompt = get_prefix_sum_prompt()
        self.worker_model = worker_model
        self.manager_model = manager_model 
        self.max_tokens_worker = max_tokens_worker
        self.max_tokens_manager = max_tokens_manager
        self.worker_prompt = worker_prompt or default_worker_prompt
        self.manager_prompt = manager_prompt or default_manager_prompt

        
    def hierarchical_process(self, input_text: str, query: str, extraction_func=None) -> str:
        """
        Hierarchical processing of input using binary tree agent structure (like prefix sum).

        Args:
            input_text: Binary string of 0s and 1s
            query: Query for each agent
            extraction_func: Optional function to extract/transform output at each step

        Returns:
            str: Final synthesized output
        """
        logging.info(f"Starting hierarchical_process with input: {input_text} and query: '{query}'")

        level_outputs = []
        for i, token in enumerate(input_text.replace(" ", "")):
            worker = WorkerAgent(self.worker_model, self.worker_prompt, max_tokens=self.max_tokens_worker)
            #print("Processing token:", token)  # Debug log
            output = worker.process_chunk(token, query, None)
            logging.info(f"Worker {i} output: {output}")
            if extraction_func:
                output = extraction_func(output)
                logging.info(f"Worker {i} extracted output: {output}")
            level_outputs.append(output)
            del worker  # Clean up worker instance

        round_num = 0
        while len(level_outputs) > 1:
            manager = ManagerAgent(self.manager_model, self.manager_prompt, max_tokens=self.max_tokens_manager)
            logging.info(f"Manager round {round_num} with {len(level_outputs)} inputs")
            next_level = []
            for i in range(0, len(level_outputs), 2):
                synthesized = manager.synthesize([level_outputs[i], level_outputs[i+1]], query)
                logging.info(f"Manager input: [{level_outputs[i]}, {level_outputs[i+1]}] -> {synthesized}")
                if extraction_func:
                    synthesized = extraction_func(synthesized)
                    logging.info(f"Extracted synthesized output: {synthesized}")
                next_level.append(synthesized)
            level_outputs = next_level
            del manager  # Clean up manager instance
            round_num += 1

        logging.info(f"Final output: {level_outputs[0]}")
        return level_outputs[0]





if __name__ == "__main__":
    # Example usage
    seq_len = 32 
    input_text = ' '.join(random.choice(['0', '1']) for _ in range(seq_len))
    maj_vote = MajorityVotingAgents(num_agents=5, max_tokens=2048)
    query = "What is the parity of the given binary string?"

    result = maj_vote.process(input_text, query)
    print(f"Majority Voting Result: {result}")
    # This will print the most common response from the agents

    # Ground truth result
    parity = "even" if input_text.count('1') % 2 == 0 else "odd"
    print(f"Ground truth parity: {parity}")
    print(f"Number of 1s in input: {input_text.count('1')}")
    
    # Example for prefix sum agents
    parity_worker_prompt, parity_manager_prompt = get_parity_prompt()
    coa = ChainOfAgents(
        worker_model="lgai/exaone-3-5-32b-instruct",
        manager_model="lgai/exaone-3-5-32b-instruct",
        chunk_size=10,  # Reduced chunk size for better handling
        worker_prompt=parity_worker_prompt,
        manager_prompt=parity_manager_prompt,
        max_tokens_worker=1024
    )

    final_output = coa.process(input_text, query, extraction_func=extract_answer)
 
    print(f"Final synthesized output: {final_output}")

    # Example for hierarchical processing
    prefix_sum_agents = PrefixSumAgents(
        worker_model="lgai/exaone-3-5-32b-instruct",
        manager_model="lgai/exaone-3-5-32b-instruct",
        max_tokens_worker=64,
        max_tokens_manager=64
    )
    final_output = prefix_sum_agents.hierarchical_process(input_text, query, extraction_func=extract_answer)
    print(f"Hierarchical processing result: {final_output}")

    # Ground truth result
    parity = "even" if input_text.count('1') % 2 == 0 else "odd"
    print(f"Ground truth parity: {parity}")
    print(f"Number of 1s in input: {input_text.count('1')}")