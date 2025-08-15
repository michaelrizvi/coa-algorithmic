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
    
    def process(self, input_text: str, query: str, extraction_func = None ) -> Dict:
        """
        Process a long text input using the Chain of Agents.
        
        Args:
            input_text: The long input text to process
            query: The user's query about the text
            
        Returns:
            Dict: Dictionary containing final response and token usage statistics
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

        if worker_token_usages:
            avg_worker_tokens = sum(worker_token_usages) / len(worker_token_usages)
            max_worker_tokens = max(worker_token_usages)
            avg_worker_prompt_tokens = sum(worker_prompt_tokens) / len(worker_prompt_tokens)
            max_worker_prompt_tokens = max(worker_prompt_tokens)

        # Synthesize results with manager agent
        manager = ManagerAgent(self.manager_model, self.manager_prompt, max_tokens=self.max_tokens_manager)
        manager_response = manager.synthesize(worker_outputs, query)
        if extraction_func:
            final_output = extraction_func(manager_response["content"])

        manager_completion_tokens = manager_response["usage"].completion_tokens
        manager_prompt_tokens = getattr(manager_response["usage"], 'prompt_tokens', 0)
        
        # Calculate total token usage (workers + manager)
        total_avg_completion_tokens = avg_worker_tokens + manager_completion_tokens if worker_token_usages else manager_completion_tokens
        total_max_completion_tokens = max_worker_tokens + manager_completion_tokens if worker_token_usages else manager_completion_tokens
        total_avg_prompt_tokens = avg_worker_prompt_tokens + manager_prompt_tokens if worker_prompt_tokens else manager_prompt_tokens
        total_max_prompt_tokens = max_worker_prompt_tokens + manager_prompt_tokens if worker_prompt_tokens else manager_prompt_tokens
        
        return {
            'content': final_output,
            'token_usage': {
                'avg_completion_tokens': total_avg_completion_tokens,
                'max_completion_tokens': total_max_completion_tokens,
                'avg_prompt_tokens': total_avg_prompt_tokens,
                'max_prompt_tokens': total_max_prompt_tokens
            }
        } 
    
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
    
    def process(self, input_text: str, query: str, extraction_func=None) -> Dict:
        agent_answers = []
        token_usages = []
        prompt_token_usages = []

        for agent_num in range(self.num_agents):
            logger.info(f"Running agent {agent_num+1}/{self.num_agents}")
            
            worker = WorkerAgent(self.model, self.prompt, max_tokens=self.max_tokens)
            result = worker.process_chunk(input_text, query, None)
            
            if extraction_func:
                answer = extraction_func(result["content"])
                if answer:
                    agent_answers.append(answer)
                else:
                    logger.warning(f"Agent {agent_num+1} returned no valid answer")
            else:
                # If no extraction function provided, use raw content
                agent_answers.append(result["content"])
            
            completion_tokens = result["usage"].completion_tokens
            prompt_tokens = getattr(result["usage"], 'prompt_tokens', 0)
            token_usages.append(completion_tokens)
            prompt_token_usages.append(prompt_tokens)
            
            del worker

        # Calculate token stats (but don't log individually)
        if token_usages:
            avg_tokens = sum(token_usages) / len(token_usages)
            max_tokens = max(token_usages)
            avg_prompt_tokens = sum(prompt_token_usages) / len(prompt_token_usages)
            max_prompt_tokens = max(prompt_token_usages)

        # Handle case where no valid answers were extracted
        if not agent_answers:
            logger.warning("No valid answers extracted from any agent")
            return {
                'content': "unknown",
                'token_usage': {
                    'avg_completion_tokens': avg_tokens if token_usages else 0,
                    'max_completion_tokens': max_tokens if token_usages else 0,
                    'avg_prompt_tokens': avg_prompt_tokens if prompt_token_usages else 0,
                    'max_prompt_tokens': max_prompt_tokens if prompt_token_usages else 0
                }
            }

        answers = {}
        for output in agent_answers:
            answers[output] = answers.get(output, 0) + 1

        most_common_answer = max(answers.items(), key=lambda x: x[1])[0]
        
        return {
            'content': most_common_answer,
            'token_usage': {
                'avg_completion_tokens': avg_tokens if token_usages else 0,
                'max_completion_tokens': max_tokens if token_usages else 0,
                'avg_prompt_tokens': avg_prompt_tokens if prompt_token_usages else 0,
                'max_prompt_tokens': max_prompt_tokens if prompt_token_usages else 0
            }
        }



class IterativeQueryAgents:
    """Class for implementing iterative hop-by-hop query processing using multiple worker agents."""
    
    def __init__(
        self,
        worker_model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        manager_model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", 
        facts_per_worker: int = 5,
        max_tokens_worker: int = 512,
        max_tokens_manager: int = 1024,
        worker_prompt: Optional[str] = None,
        manager_prompt: Optional[str] = None,
        num_hops: int = 2
    ):
        """
        Initialize the Iterative Query Agents.
        
        Args:
            worker_model: Model to use for worker agents
            manager_model: Model to use for manager agent
            facts_per_worker: Number of facts per worker agent
            max_tokens_worker: Max tokens for worker responses
            max_tokens_manager: Max tokens for manager responses
            worker_prompt: Custom system prompt for workers
            manager_prompt: Custom system prompt for manager
            num_hops: Number of hops in the reasoning chain
        """
        self.worker_model = worker_model
        self.manager_model = manager_model
        self.facts_per_worker = facts_per_worker
        self.max_tokens_worker = max_tokens_worker
        self.max_tokens_manager = max_tokens_manager
        self.num_hops = num_hops
        
        # Default prompts
        default_worker_prompt = """You are a helpful assistant that answers questions based ONLY on the given facts.

IMPORTANT: You have been given only a small subset of all available facts. It is very likely that the fact needed to answer the query is NOT in your subset.

You will be given:
1. A limited set of facts about relationships between people
2. A specific query about one relationship

Instructions:
- ONLY look through the facts provided to you
- If you find the EXACT fact needed to answer the query, extract the answer
- If the exact fact is NOT in your subset (which is very common), respond with "Not Found"
- DO NOT guess or infer answers from similar facts
- DO NOT make assumptions about relationships not explicitly stated
- Always format your response as: Answer: [YourAnswer]

Example (found):
Facts: "John's boss is Mary. Alice's teacher is Bob."
Query: "Who is John's boss?"
Response: Answer: Mary

Example (not found - very common):
Facts: "John's boss is Mary. Alice's teacher is Bob."
Query: "Who is Sarah's mentor?"
Response: Answer: Not Found

Example (not found - don't guess):
Facts: "John's boss is Mary. Alice's teacher is Bob."
Query: "Who is Mary's supervisor?"  
Response: Answer: Not Found

Remember: Most queries will not have their answer in your subset of facts. Only answer if the exact fact is present."""

        default_manager_prompt = """You are a manager agent that coordinates multi-hop reasoning queries.

Your task is to take an answer from a previous query and generate the next query in the reasoning chain.

You will be given:
1. The original multi-hop question
2. The current intermediate answer
3. The current step number

Instructions:
- Use the intermediate answer to construct the next query
- Format your response as: Next Query: [YourQuery]

Example:
Original question: "Who is the supervisor of the boss of John?"
Current answer: "Mary" (John's boss)
Response: Next Query: Who is Mary's supervisor?"""

        self.worker_prompt = worker_prompt or default_worker_prompt
        self.manager_prompt = manager_prompt or default_manager_prompt
        
        logger.info(f"Initialized Iterative Query Agents with {self.facts_per_worker} facts per worker, {self.num_hops} hops")

    def _split_facts_into_chunks(self, facts_string: str) -> List[str]:
        """Split facts string into chunks for worker agents."""
        # Split facts by sentence (assuming facts are separated by ". ")
        facts_list = [fact.strip() for fact in facts_string.split(". ") if fact.strip()]
        
        # Group facts into chunks
        chunks = []
        for i in range(0, len(facts_list), self.facts_per_worker):
            chunk_facts = facts_list[i:i + self.facts_per_worker]
            chunk_string = ". ".join(chunk_facts)
            if not chunk_string.endswith('.'):
                chunk_string += '.'
            chunks.append(chunk_string)
        
        return chunks

    def _extract_answer(self, response: str) -> str:
        """Extract answer from worker response."""
        pattern = r"Answer:\s*(.+?)(?:\n|$)"
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""

    def _extract_next_query(self, response: str) -> str:
        """Extract next query from manager response."""
        pattern = r"Next Query:\s*(.+?)(?:\n|$)"
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""

    def _parse_original_query(self, query: str) -> Tuple[List[str], str]:
        """Parse the original query to extract the sequence of relations."""
        # For query like "Who is the supervisor of the boss of John?"
        # Extract: ["boss", "supervisor"] and base entity "John"
        
        # Remove "Who is the" and "?"
        clean_query = query.replace("Who is the", "").replace("?", "").strip()
        
        # Split by " of " to get parts
        parts = clean_query.split(" of ")
        
        relations = []
        base_entity = parts[-1].strip()  # Last part is the base entity
        
        # Extract relations from all parts except the last (base entity)
        for i in range(len(parts) - 1):
            part = parts[i].strip()
            # Remove "the" prefix if present
            if part.startswith("the "):
                part = part[4:]
            relations.append(part)
        
        # Reverse the relations list because we need to process them from inside out
        # "supervisor of the boss of John" -> first find John's boss, then boss's supervisor
        relations.reverse()
        
        return relations, base_entity

    def process(self, input_text: str, query: str, extraction_func=None) -> Dict:
        """
        Process a K-hop query using iterative worker agents.
        
        Args:
            input_text: String containing all facts
            query: The multi-hop query to answer
            extraction_func: Function to extract final answer (optional)
            
        Returns:
            Dict: Dictionary containing final answer and token usage statistics
        """
        # Parse the original query to understand the hop sequence
        relations, current_entity = self._parse_original_query(query)
        logger.info(f"Parsed query - Relations: {relations}, Base entity: {current_entity}")
        
        # Split facts into chunks for workers
        fact_chunks = self._split_facts_into_chunks(input_text)
        logger.info(f"Split facts into {len(fact_chunks)} chunks")
        
        # Track token usage
        total_worker_tokens = 0
        total_manager_tokens = 0
        total_worker_prompt_tokens = 0
        total_manager_prompt_tokens = 0
        
        # Iterate through each hop
        for hop_idx in range(self.num_hops):
            if hop_idx < len(relations):
                # Construct query for this hop
                current_query = f"Who is {current_entity}'s {relations[hop_idx]}?"
            else:
                logger.warning(f"Not enough relations for hop {hop_idx}")
                break
                
            logger.info(f"Hop {hop_idx + 1}: {current_query}")
            
            # Query all worker agents
            worker_answers = []
            hop_worker_tokens = []
            hop_worker_prompt_tokens = []
            
            for i, chunk in enumerate(fact_chunks):
                worker = WorkerAgent(self.worker_model, self.worker_prompt, max_tokens=self.max_tokens_worker)
                response = worker.process_chunk(chunk, current_query)
                
                answer = self._extract_answer(response["content"])
                worker_answers.append(answer)
                
                hop_worker_tokens.append(response["usage"].completion_tokens)
                hop_worker_prompt_tokens.append(getattr(response["usage"], 'prompt_tokens', 0))
                
                logger.info(f"Worker {i+1} answer: {answer}")
                del worker
            
            # Find the first valid answer (not "Not Found")
            current_answer = ""
            for answer in worker_answers:
                if answer and answer.lower() != "not found":
                    current_answer = answer
                    break
            
            if not current_answer:
                logger.warning(f"No worker found answer for hop {hop_idx + 1}")
                return {
                    'content': "",
                    'token_usage': {
                        'avg_completion_tokens': 0,
                        'max_completion_tokens': 0,
                        'avg_prompt_tokens': 0,
                        'max_prompt_tokens': 0
                    }
                }
            
            # Update token tracking
            total_worker_tokens += sum(hop_worker_tokens)
            total_worker_prompt_tokens += sum(hop_worker_prompt_tokens)
            
            logger.info(f"Found answer for hop {hop_idx + 1}: {current_answer}")
            
            # If this is the last hop, return the answer
            if hop_idx == self.num_hops - 1:
                final_answer = current_answer
                if extraction_func:
                    final_answer = extraction_func(current_answer)
                
                # Calculate average token usage
                avg_completion_tokens = total_worker_tokens / max(1, len(fact_chunks) * self.num_hops)
                max_completion_tokens = max(hop_worker_tokens) if hop_worker_tokens else 0
                avg_prompt_tokens = total_worker_prompt_tokens / max(1, len(fact_chunks) * self.num_hops)
                max_prompt_tokens = max(hop_worker_prompt_tokens) if hop_worker_prompt_tokens else 0
                
                return {
                    'content': final_answer,
                    'token_usage': {
                        'avg_completion_tokens': avg_completion_tokens + total_manager_tokens,
                        'max_completion_tokens': max_completion_tokens + total_manager_tokens,
                        'avg_prompt_tokens': avg_prompt_tokens + total_manager_prompt_tokens,
                        'max_prompt_tokens': max_prompt_tokens + total_manager_prompt_tokens
                    }
                }
            
            # Use manager to generate next query
            manager = ManagerAgent(self.manager_model, self.manager_prompt, max_tokens=self.max_tokens_manager)
            manager_input = f"Original question: {query}\nCurrent answer: {current_answer}\nStep: {hop_idx + 1}"
            manager_response = manager.synthesize([manager_input], "Generate the next query")
            
            total_manager_tokens += manager_response["usage"].completion_tokens
            total_manager_prompt_tokens += getattr(manager_response["usage"], 'prompt_tokens', 0)
            
            # Extract next query (but we don't actually use it since we parse the original query)
            # This is for completeness and future flexibility
            next_query = self._extract_next_query(manager_response["content"])
            logger.info(f"Manager suggested next query: {next_query}")
            
            # Update current entity for next hop
            current_entity = current_answer
            del manager
        
        # Should not reach here if num_hops is set correctly
        logger.warning("Completed all hops without returning")
        return {
            'content': "",
            'token_usage': {
                'avg_completion_tokens': 0,
                'max_completion_tokens': 0,
                'avg_prompt_tokens': 0,
                'max_prompt_tokens': 0
            }
        }


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

    def hierarchical_process(self, input_text: str, query: str, extraction_func=None) -> Dict:
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
        all_completion_tokens = []
        all_prompt_tokens = []
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
                
                # Collect individual agent token usage
                all_completion_tokens.append(completion_tokens)
                all_prompt_tokens.append(prompt_tokens)
                
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
            
        # Calculate individual agent token usage statistics
        mean_completion_tokens = 0
        max_completion_tokens_single = 0
        mean_prompt_tokens = 0
        max_prompt_tokens_single = 0
        
        if all_completion_tokens:
            mean_completion_tokens = sum(all_completion_tokens) / len(all_completion_tokens)
            max_completion_tokens_single = max(all_completion_tokens)
            logger.info(f"Mean completion tokens per agent: {mean_completion_tokens:.2f}")
            logger.info(f"Max completion tokens per agent: {max_completion_tokens_single}")
            
        if all_prompt_tokens:
            mean_prompt_tokens = sum(all_prompt_tokens) / len(all_prompt_tokens)
            max_prompt_tokens_single = max(all_prompt_tokens)
            logger.info(f"Mean prompt tokens per agent: {mean_prompt_tokens:.2f}")
            logger.info(f"Max prompt tokens per agent: {max_prompt_tokens_single}")

        logging.info(f"Final output: {level_outputs[0]}")
        
        return {
            'content': level_outputs[0],
            'token_usage': {
                'avg_completion_tokens': sum_of_avg_completion_tokens,
                'max_completion_tokens': sum_of_max_completion_tokens,
                'avg_prompt_tokens': sum_of_avg_prompt_tokens,
                'max_prompt_tokens': sum_of_max_prompt_tokens,
                'mean_completion_tokens_per_agent': mean_completion_tokens,
                'max_completion_tokens_per_agent': max_completion_tokens_single,
                'mean_prompt_tokens_per_agent': mean_prompt_tokens,
                'max_prompt_tokens_per_agent': max_prompt_tokens_single
            }
        }


def test_majority_voting(query, input_text):
    maj_vote = MajorityVotingAgents(num_agents=5, max_tokens=2048)
    result = maj_vote.process(input_text, query)
    print(f"Majority Voting Result: {result['content']}")
    print(f"Token Usage: {result['token_usage']}")

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
    result = coa.process(input_text, query, extraction_func=extract_answer)
    print(f"Chain-of-Agents Result: {result['content']}")
    print(f"Token Usage: {result['token_usage']}")

def test_prefix_sum(query, input_text):
    prefix_sum_agents = PrefixSumAgents(
        worker_model="lgai/exaone-3-5-32b-instruct",
        manager_model="lgai/exaone-3-5-32b-instruct",
        max_tokens_worker=64,
        max_tokens_manager=64
    )
    result = prefix_sum_agents.hierarchical_process(input_text, query, extraction_func=extract_answer)
    print(f"Prefix Sum Result: {result['content']}")
    print(f"Token Usage: {result['token_usage']}")

def test_iterative_query(facts, query, num_hops):
    iterative_agents = IterativeQueryAgents(
        worker_model="lgai/exaone-3-5-32b-instruct",
        manager_model="lgai/exaone-3-5-32b-instruct",
        facts_per_worker=3,
        max_tokens_worker=256,
        max_tokens_manager=256,
        num_hops=num_hops
    )
    result = iterative_agents.process(facts, query)
    print(f"Iterative Query Result: {result['content']}")
    print(f"Token Usage: {result['token_usage']}")

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
