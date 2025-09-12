from typing import List, Dict, Tuple
import logging
import fitz  # PyMuPDF
import re
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INDEX_HINT_STRING ="""
    You will receive a binary string where each bit is preceded by its 1-based index in the format: [i=INDEX] BIT. For example, the string:

    [i=1] 1 [i=2] 0 [i=3] 1

    represents the binary sequence 101. Use the index and bit values as needed for your task (e.g., computing parity or identifying bit positions).
    """

def read_pdf(pdf_path: str) -> str:
    """
    Read text content from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        str: Extracted text from the PDF
    """
    try:
        text = []
        with fitz.open(pdf_path) as doc:
            logger.info(f"Processing PDF with {len(doc)} pages")
            for page in doc:
                text.append(page.get_text())
        
        return "\n".join(filter(None, text))  # Filter out empty strings
    except Exception as e:
        logger.error(f"Error reading PDF: {str(e)}")
        raise

def split_into_chunks(text: str, chunk_size: int, model: str = "llama-3.3-70b-versatile") -> List[str]:
    """
    Split text into chunks based on word count.
    
    Args:
        text: The input text to split
        chunk_size: Maximum number of words per chunk
        model: Not used, kept for compatibility
        
    Returns:
        List[str]: List of text chunks
    """
    # Split by paragraphs first to maintain context
    paragraphs = text.split('\n\n')
    words = []
    current_chunk = []
    chunks = []
    
    for paragraph in paragraphs:
        # Skip empty paragraphs
        if not paragraph.strip():
            continue
            
        paragraph_words = paragraph.split()
        
        # If adding this paragraph exceeds chunk size, save current chunk and start new one
        if len(current_chunk) + len(paragraph_words) > chunk_size:
            if current_chunk:  # Save current chunk if it exists
                chunks.append(' '.join(current_chunk))
                current_chunk = []
            
            # Handle paragraphs larger than chunk_size
            while len(paragraph_words) > chunk_size:
                chunks.append(' '.join(paragraph_words[:chunk_size]))
                paragraph_words = paragraph_words[chunk_size:]
            
            current_chunk = paragraph_words
        else:
            current_chunk.extend(paragraph_words)
    
    # Add any remaining text
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    logger.info(f"Split text into {len(chunks)} chunks")
    return chunks

def count_tokens(text: str, model: str = "llama-3.3-70b-versatile") -> int:
    """
    Count the number of words in a text string.
    
    Args:
        text: The input text
        model: Not used, kept for compatibility
        
    Returns:
        int: Number of words
    """
    return len(text.split())

def get_default_prompts() -> tuple[str, str]:
    """
    Get default system prompts for worker and manager agents.
    
    Returns:
        tuple[str, str]: (worker_prompt, manager_prompt)
    """
    worker_prompt = """You are a worker agent responsible for analyzing a portion of a document.
Your task is to identify key information related to the user's query and provide clear, concise analysis."""

    manager_prompt = """You are a manager agent responsible for synthesizing information from multiple workers.
Your task is to combine their analyses into a coherent, comprehensive response that directly answers the user's query."""

    return worker_prompt, manager_prompt 

def get_majority_vote_prompt(index_hints: bool=False) -> str:
    """
    Get the system prompt for majority vote synthesis.
    
    Returns:
        str: The majority vote synthesis prompt
    """
    prompt = """You are a reasoning agent responsible for analyzing a portion of a document.
    Your task is to provide an analysis of the binary string provided in your chunk and determine if it is even or odd parity. To compute the parity, follow these steps:
    1. Count the number of 1's in the binary string.
    2. If the count is even, return 0.
    3. If the count is odd, return 1.
    4. Present the final answer in the format "The answer is: [your answer]"
    
    You MUST use the following template. Here is an example for "1011":
    1: 1 (count: 1)
    2: 0 (count: 1)
    3: 1 (count: 2)
    4: 1 (count: 3)
    Final count: 3
    The answer is: 1
    """
    if index_hints:
        prompt += INDEX_HINT_STRING
    return prompt

def get_prefix_sum_prompt(index_hints: bool=False, b: int = 2) -> str:
    """
    Get the system prompt for prefix sum calculation.
    
    Returns:
        tuple[str, str]: The prefix sum prompt
    """
    worker_prompt = """You are a worker agent responsible for calculating the parity of a single binary digit. if the digit is 1, you will return 1, otherwise you will return 0.
    Present the final answer in the format "The answer is: [your answer]"
    
    IMPORTANT:
    - Be concise and direct in your response. You MUST think by steps, but do not repeat these instructions in your output.
    - For long lists with many 1s, carefully double-check your counting to avoid arithmetic errors.
    """
    manager_prompt = f"""You are a manager agent responsible for synthesizing the results of previous workers.
    Your task is to return the parity of the binary string provided by the worker agents. You may think step by step, but your final answer should be concise and clear.
    To compute the parity, follow these steps:
    1. Collect the results from the worker agents. This should be a list of binary digits (0 or 1).
    2. If the parity of the list is even, return 0.
    3. If the parity of the list is odd, return 1.
    5. Present the final answer on a new line in the format "The answer is: [your answer]" 
    
    IMPORTANT: Show your work step by step to demonstrate thorough analysis:
    1. Go through each bit position and note its value
    2. Keep a running count of 1s encountered
    3. State the final count
    4. Determine if the count is even or odd

    You MUST use the following template. Here is an example for "1011":
    1: 1
    0: 1
    1: 2
    1: 3
    Final count: 3
    The answer is: 1
    """
    if index_hints:
        worker_prompt += INDEX_HINT_STRING
        manager_prompt += INDEX_HINT_STRING
    return worker_prompt, manager_prompt

def get_parity_prompt(index_hints: bool=False) -> str:
    """
    Get the system prompt for parity calculation.
    
    Returns:
        str: The parity calculation prompt
    """
    parity_worker_prompt = """You are a worker agent responsible for analyzing a portion of a document.
    Your task is to provide an analysis of the binary string provided in your chunk and determine if it is even or odd parity. To compute the parity, follow these steps:
    1. Count the number of 1's in the binary string.
    2. If the count is even, return 0.
    3. If the count is odd, return 1.
    4. Provide your result in a clear and concise manner.
    5. Present the final answer in the format "The answer is: [your answer]"
    
    You MUST use the following template. Here is an example for "1011":
    1: 1
    0: 1
    1: 2
    1: 3
    Final count: 3
    The answer is: 1
    """

    parity_manager_prompt = """You are a manager agent responsible for synthesizing information from multiple workers.
    Your task is to combine their provided parities and determine the overall parity of the binary string. To compute the aggregate parity, follow these steps:
    1. Collect the parity results from all worker agents.
    2. Each worker will return either 0 or 1.
    3. Count the number of 1 responses.
    4. If the count of 1 responses is even, the overall parity is 0.
    5. If the count of 1 responses is odd, the overall parity is 1.
    6. Present the final answer in the format "The answer is: [your answer]"
    
    You MUST use the following template. Here is an example for "1011":
    1: 1
    0: 1
    1: 2
    1: 3
    Final count: 3
    The answer is: 1
    """
    if index_hints:
        parity_worker_prompt += INDEX_HINT_STRING
        parity_manager_prompt += INDEX_HINT_STRING
    return parity_worker_prompt, parity_manager_prompt

def extract_answer(text):
    match = re.search(r"The answer is:?\s*(.+?)(?:\\|\n|$)", text, re.IGNORECASE)
    return match.group(1).strip() if match else None

def split_binary_string(s, chunk_size: int) -> List[str]:
    entries = re.findall(r"\[i=\d+\] \d", s)
    chunks = [
        " ".join(entries[i:i + chunk_size])
        for i in range(0, len(entries), chunk_size)
    ]
    return chunks

def generate_bitstring(length, index_hints=False):
    """Generate a random binary string of given length.
    If index_hints is True, the string will be generated with hints for parity calculation.
    """
    if index_hints:
        return ' '.join(f'[i={i+1}] {random.choice("01")}' for i in range(length))
    return ''.join(random.choice('01') for _ in range(length))

def compute_parity(bitstring):
    return str(bitstring.count('1') % 2)

def get_permutation_majority_vote_prompt() -> str:
    """
    Get the system prompt for majority vote permutation solving.
    
    Returns:
        str: The majority vote permutation prompt
    """
    return """You are a reasoning agent responsible for tracking ball positions through a sequence of swaps.

Your task is to determine the final position of each ball after performing all the given swap operations.

Initial state: Each ball starts in its corresponding bin (ball 1 in bin 1, ball 2 in bin 2, etc.).

To solve this problem:
1. First, identify which balls are mentioned in the swap operations - ONLY track these balls
2. Start with balls in their initial positions (e.g., if balls 1, 2, 3 are mentioned: {1:1, 2:2, 3:3})
3. For each swap operation "Swap ball X and ball Y":
   - Find the current bins of ball X and ball Y
   - Exchange their positions
4. Continue until all swaps are processed
5. Present your final answer as a dictionary mapping ONLY the balls mentioned in swaps to their final positions

IMPORTANT: Only include balls that appear in the swap operations. Do not add extra balls.

Present the final answer in the format "The answer is: {ball1:bin1, ball2:bin2, ...}" for only the balls involved in swaps.
"""

def get_permutation_prompts() -> tuple[str, str]:
    """
    Get the system prompts for permutation worker and manager agents.
    
    Returns:
        tuple[str, str]: (worker_prompt, manager_prompt)
    """
    worker_prompt = """You are a worker agent responsible for processing a portion of swap operations in a larger sequence.

Your task is to carefully track ball positions through your assigned swap operations and report the precise current state.

You will receive:
- Current ball positions as a dictionary (e.g., {1:2, 2:1, 3:3})
- A sequence of swap operations to process

Instructions:
1. Start with the EXACT positions given to you - this is the state after previous swaps
2. Process each swap operation "Swap ball X and ball Y" in order:
   - Find the current bins of ball X and ball Y
   - Exchange ONLY their positions
   - Keep all other balls in their current positions
3. Track each swap carefully - one mistake will affect the final result
4. Report the state after processing ALL your assigned swaps

CRITICAL: Only include the balls that are present in the input positions. The exact same balls, no more, no less.

Present the final answer in the format "The answer is: {ball1:bin1, ball2:bin2, ...}" with the exact same ball numbers as your input.
"""

    manager_prompt = """You are a manager agent responsible for determining the final ball positions from worker results.

Your task is to identify the final state of all balls after all swap operations have been processed by your workers.

You will receive position dictionaries from multiple workers who processed different portions of the swap sequence in order. The workers processed swaps sequentially, so:

Instructions:
1. The workers processed swaps in chronological order (worker 1 → worker 2 → worker 3, etc.)
2. Each worker started with the positions left by the previous worker
3. The LAST worker's result contains the final positions after all swaps
4. Simply report the last worker's position dictionary as the final answer

CRITICAL: Take the position dictionary from the last (final) worker only. This represents the complete final state.

Present the final answer in the format "The answer is: {ball1:bin1, ball2:bin2, ...}" exactly as reported by the final worker.
"""

    return worker_prompt, manager_prompt

def get_permutation_prefix_sum_prompts(b: int = 2) -> tuple[str, str]:
    """
    Get the system prompts for prefix sum permutation calculation.
    
    Args:
        b: Branching factor for hierarchical processing
        
    Returns:
        tuple[str, str]: (worker_prompt, manager_prompt)
    """
    worker_prompt = """You are a worker agent in a hierarchical system processing ONE swap operation.

Your task is to apply exactly one swap operation and report the resulting ball positions with perfect accuracy.

You will receive:
- Current ball positions as a dictionary (e.g., {1:3, 2:1, 3:2})
- ONE swap operation: "Swap ball X and ball Y"

Process the swap step-by-step:
1. Identify ball X's current bin and ball Y's current bin from the dictionary
2. Exchange ONLY these two positions:
   - Ball X goes to the bin where ball Y was
   - Ball Y goes to the bin where ball X was
3. ALL other balls stay in their exact same positions
4. Output the complete updated dictionary

Example:
- Input positions: {1:3, 2:1, 3:2}
- Swap operation: "Swap ball 1 and ball 3"
- Ball 1 is in bin 3, ball 3 is in bin 2
- After swap: {1:2, 2:1, 3:3}

CRITICAL: Include ALL balls from input with exact same ball numbers. One swap affects exactly two positions.

Present the final answer in the format "The answer is: {ball1:bin1, ball2:bin2, ...}" with all balls from your input.
"""

    manager_prompt = f"""You are a manager agent in a hierarchical ball-tracking system combining results from {b} workers.

Your task is to determine the final ball positions after your workers processed their assigned swaps in sequence.

You will receive position dictionaries from {b} workers who processed swaps in chronological order. Each worker:
- Started with the ball positions left by the previous worker
- Applied exactly one swap operation
- Reported the updated positions

Your job is simple:
1. The workers processed swaps sequentially (worker 1 → worker 2 → ... → worker {b})
2. The LAST worker's position dictionary represents the final state after all {b} swaps
3. Report the last worker's result as your output

Example with 2 workers:
- Worker 1 result: {{1:2, 2:3, 3:1}} (after 1st swap)
- Worker 2 result: {{1:3, 2:2, 3:1}} (after 2nd swap) ← This is your answer

CRITICAL: Output exactly the position dictionary from the final (last) worker. This contains the cumulative effect of all swaps.

Present the final answer in the format "The answer is: {{ball1:bin1, ball2:bin2, ...}}" exactly as reported by the last worker.
"""

    return worker_prompt, manager_prompt

def extract_position_dict(text: str) -> str:
    """
    Extract position dictionary from agent response text.
    
    Args:
        text: Agent response text containing position dictionary
        
    Returns:
        str: Extracted dictionary string, or None if not found
    """
    import re
    # Look for dictionary pattern in the text
    match = re.search(r"The answer is:?\s*(\{[^}]+\})", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Fallback: look for any dictionary-like pattern
    match = re.search(r"\{[^}]+\}", text)
    if match:
        return match.group(0).strip()
    
    return None

def parse_position_dict(dict_str: str) -> dict:
    """
    Parse position dictionary string to actual dictionary.
    
    Args:
        dict_str: String representation of position dictionary
        
    Returns:
        dict: Parsed position dictionary
    """
    try:
        # Clean up the string and evaluate it
        dict_str = dict_str.replace(" ", "").replace("'", "").replace('"', '')
        return eval(dict_str)
    except:
        return None

def evaluate_permutation_accuracy(predicted_positions: dict, true_positions: list) -> tuple[bool, float]:
    """
    Evaluate permutation prediction accuracy.
    
    Args:
        predicted_positions: Dictionary mapping ball -> bin position
        true_positions: List where true_positions[i] is the bin for ball i+1
        
    Returns:
        tuple[bool, float]: (exact_match, element_accuracy)
    """
    if predicted_positions is None:
        return False, 0.0
    
    # Convert true_positions list to dict format
    true_dict = {i+1: true_positions[i] for i in range(len(true_positions))}
    
    # Check exact match (both dictionaries must be identical)
    exact_match = predicted_positions == true_dict
    
    # Calculate element-wise accuracy
    correct_elements = 0
    total_elements = len(true_positions)
    
    for ball in range(1, total_elements + 1):
        if ball in predicted_positions and predicted_positions[ball] == true_dict[ball]:
            correct_elements += 1
    
    element_accuracy = correct_elements / total_elements if total_elements > 0 else 0.0
    
    # Debug logging for dictionary composition issues
    if element_accuracy == 1.0 and not exact_match:
        print(f"DEBUG: Element accuracy is 1.0 but exact match is False!")
        print(f"Predicted: {predicted_positions}")
        print(f"True dict: {true_dict}")
        print(f"Keys match: {set(predicted_positions.keys()) == set(true_dict.keys())}")
    
    # Additional debugging for composition issues
    if predicted_positions is not None and len(predicted_positions) != len(true_dict):
        print(f"DEBUG: Dictionary size mismatch - Pred: {len(predicted_positions)}, True: {len(true_dict)}")
        print(f"Predicted keys: {sorted(predicted_positions.keys())}")
        print(f"True keys: {sorted(true_dict.keys())}")
    
    return exact_match, element_accuracy

def generate_permutation_problem(n: int = 5, num_swaps: int = None) -> tuple[str, List[int]]:
    """
    Generate an Sn permutation problem in natural language.
    
    The problem starts with balls 1-n in bins 1-n respectively, then performs
    a series of swaps to create a permutation. Returns both the natural language
    description and the final positions array.
    
    Args:
        n: Number of elements in the permutation group (default 5 for S5)
        num_swaps: Number of swaps to perform. If None, uses random number between n//2 and 2*n
        
    Returns:
        tuple[str, List[int]]: (problem_description, final_positions)
            where final_positions[i] is the bin number where ball i+1 ends up
    """
    if num_swaps is None:
        num_swaps = random.randint(max(1, n//2), 2*n)
    
    # Initialize positions: ball i is in bin i (1-indexed)
    positions = list(range(1, n+1))  # positions[i] = bin number for ball i+1
    
    # Generate random swaps
    swaps = []
    for _ in range(num_swaps):
        # Choose two different balls to swap
        ball1, ball2 = random.sample(range(1, n+1), 2)
        swaps.append((ball1, ball2))
        
        # Apply the swap to track actual positions
        # Find current bins of the two balls
        bin1 = positions[ball1 - 1]
        bin2 = positions[ball2 - 1] 
        
        # Swap their positions
        positions[ball1 - 1] = bin2
        positions[ball2 - 1] = bin1
    
    # Generate swap commands with separator tokens
    problem_parts = []
    
    for i, (ball1, ball2) in enumerate(swaps):
        problem_parts.append(f"Swap ball {ball1} and ball {ball2}")
        if i < len(swaps) - 1:  # Add separator except after last swap
            problem_parts.append("|")
    
    problem_description = " ".join(problem_parts)
    
    return problem_description, positions


# K-hop task utilities

def get_khop_entities() -> List[str]:
    """Generate a list of 50 balanced single-token entity names."""
    male_names = [
        "Adam", "Ben", "Carl", "Dan", "Emil", "Frank", "Gary", "Henry", "Ian", "Jack",
        "Kevin", "Leo", "Max", "Noah", "Owen", "Paul", "Quinn", "Ryan", "Sam", "Tom",
        "Victor", "Will", "Xavier", "York", "Zane"
    ]
    
    female_names = [
        "Alice", "Beth", "Carol", "Diana", "Emma", "Fiona", "Grace", "Helen", "Iris", "Jane",
        "Kate", "Lucy", "Mary", "Nina", "Olivia", "Paula", "Rose", "Sarah", "Tina", "Uma",
        "Vera", "Wendy", "Zoe", "Ana", "Eva"
    ]
    
    return male_names + female_names


def get_khop_relations() -> List[str]:
    """Generate a list of 20 diverse single-token relations."""
    return [
        "boss", "instructor", "teacher", "advisor", "supervisor", "mentor", "coach", 
        "manager", "leader", "director", "guide", "trainer", "tutor", "professor",
        "captain", "chief", "head", "principal", "coordinator", "facilitator"
    ]


def generate_khop_problem(k: int, n: int, entities: List[str], relations: List[str]) -> Tuple[str, str, str]:
    """
    Generate a K-hop reasoning problem.
    
    Args:
        k: Number of hops needed to solve the query
        n: Total number of facts to include
        entities: List of entity names
        relations: List of relation names
        
    Returns:
        Tuple of (facts_string, query, ground_truth_answer)
    """
    if n < k:
        raise ValueError(f"Number of facts ({n}) must be at least number of hops ({k})")
    
    # Generate the chain for the query (ground truth path)
    query_entities = random.sample(entities, k + 1)  # k+1 entities for k relations
    query_relations = random.sample(relations, k)
    
    # Create the chain: entity0 -> rel0 -> entity1 -> rel1 -> entity2 -> ... -> entityK
    ground_truth_facts = []
    for i in range(k):
        fact = f"{query_entities[i]}'s {query_relations[i]} is {query_entities[i + 1]}"
        ground_truth_facts.append(fact)
    
    # Generate additional random facts (n - k facts)
    additional_facts = []
    used_pairs = set()
    
    # Track the pairs we've already used in ground truth
    for i in range(k):
        used_pairs.add((query_entities[i], query_relations[i]))
    
    remaining_entities = [e for e in entities if e not in query_entities]
    remaining_relations = [r for r in relations if r not in query_relations] + query_relations  # Can reuse relations
    
    attempts = 0
    max_attempts = 1000
    
    while len(additional_facts) < (n - k) and attempts < max_attempts:
        attempts += 1
        
        # Pick entities (can reuse entities not in query path, or use completely new ones)
        entity1 = random.choice(entities)
        entity2 = random.choice(entities)
        relation = random.choice(relations)
        
        # Avoid conflicts with ground truth and duplicates
        if (entity1, relation) not in used_pairs and entity1 != entity2:
            fact = f"{entity1}'s {relation} is {entity2}"
            if fact not in additional_facts and fact not in ground_truth_facts:
                additional_facts.append(fact)
                used_pairs.add((entity1, relation))
    
    # Combine all facts and shuffle them
    all_facts = ground_truth_facts + additional_facts
    random.shuffle(all_facts)
    
    # Create the query
    query_chain = []
    for i in range(k-1, -1, -1):  # Reverse order for query
        query_chain.append(f"the {query_relations[i]}")
    
    query = f"Who is {' of '.join(query_chain)} of {query_entities[0]}?"
    ground_truth_answer = query_entities[-1]
    
    # Format facts as a single string
    facts_string = ". ".join(all_facts) + "."
    
    return facts_string, query, ground_truth_answer


def extract_khop_answer(response: str) -> str:
    """
    Extract answer from K-hop response in format 'Answer: [EntityName]'.
    
    Args:
        response: The agent's response
        
    Returns:
        str: Extracted entity name or empty string if not found
    """
    # Look for pattern "Answer: EntityName"
    pattern = r"Answer:\s*([A-Za-z]+)"
    match = re.search(pattern, response, re.IGNORECASE)
    
    if match:
        return match.group(1).strip()
    
    # Fallback: look for any single capitalized word at the end
    words = response.strip().split()
    if words:
        last_word = words[-1].rstrip('.')
        if last_word.isalpha() and last_word[0].isupper():
            return last_word
    
    return ""


def get_khop_majority_vote_prompt() -> str:
    """Generate prompt for majority voting agents on K-hop task."""
    return """You are an expert at logical reasoning and following relationship chains. 

Your task is to answer questions about relationships between people by following chains of connections through the given facts.

You will be given:
1. A set of facts describing relationships between people (e.g., "Alice's boss is Bob")
2. A query asking about a multi-step relationship chain

Instructions:
- Read all the facts carefully
- Follow the relationship chain step by step
- Track each connection to find the final answer
- Output your answer in the exact format: Answer: [PersonName]

Example:
Facts: "John's boss is Mary. Mary's supervisor is Tom."
Query: "Who is the supervisor of the boss of John?"
Reasoning: John's boss is Mary → Mary's supervisor is Tom
Answer: Tom

Be systematic and double-check your reasoning chain."""


# List manipulation task utilities

def generate_list_manipulation_operations(num_ops: int = 7) -> List[str]:
    """
    Generate the list of available operations ordered by difficulty.
    
    Args:
        num_ops: Number of operations to include (1-7, ordered by difficulty)
        
    Returns:
        List[str]: List of operation names in order of difficulty
    """
    all_operations = [
        "swap",           # 1. Swap(k, j) - simplest
        "reverse",        # 2. Reverse - simple  
        "rotate_left",    # 3. Rotate left by k
        "rotate_right",   # 4. Rotate right by k
        "reverse_subrange", # 5. Reverse subrange
        "even_odd_indices", # 6. Even then odd indices
        "index_select"    # 7. Index-based selection - most complex
    ]
    return all_operations[:num_ops]

def apply_list_operation(lst: List[int], operation: str, **kwargs) -> List[int]:
    """
    Apply a single list manipulation operation.
    
    Args:
        lst: Input list
        operation: Operation name
        **kwargs: Operation parameters
        
    Returns:
        List[int]: Resulting list after operation
    """
    result = lst.copy()
    
    if operation == "swap":
        k, j = kwargs['k'], kwargs['j']
        result[k], result[j] = result[j], result[k]
    
    elif operation == "reverse":
        result = result[::-1]
    
    elif operation == "rotate_left":
        k = kwargs['k']
        result = result[k:] + result[:k]
    
    elif operation == "rotate_right":
        k = kwargs['k']
        result = result[-k:] + result[:-k]
    
    elif operation == "reverse_subrange":
        i, j = kwargs['i'], kwargs['j']
        result[i:j] = reversed(result[i:j])
    
    elif operation == "even_odd_indices":
        result = result[::2] + result[1::2]
    
    elif operation == "index_select":
        indices = kwargs['indices']
        result = [result[i] for i in indices]
    
    return result

def generate_operation_with_params(lst_size: int, operation: str, use_python_code: bool = False) -> tuple[str, dict]:
    """
    Generate a random operation with valid parameters.
    
    Args:
        lst_size: Size of the list
        operation: Operation name
        use_python_code: If True, return Python code strings; if False, natural language
        
    Returns:
        tuple[str, dict]: (operation_description, parameters)
    """
    if operation == "swap":
        k, j = random.sample(range(lst_size), 2)
        params = {'k': k, 'j': j}
        if use_python_code:
            desc = f"a[{k}], a[{j}] = a[{j}], a[{k}]"
        else:
            desc = f"Swap elements at positions {k} and {j}"
        
    elif operation == "reverse":
        params = {}
        if use_python_code:
            desc = "a[::-1]"
        else:
            desc = "Reverse the entire list"
        
    elif operation == "rotate_left":
        k = random.randint(1, lst_size - 1)
        params = {'k': k}
        if use_python_code:
            desc = f"a[:] = a[{k}:] + a[:{k}]"
        else:
            desc = f"Rotate left by {k} positions"
        
    elif operation == "rotate_right":
        k = random.randint(1, lst_size - 1)
        params = {'k': k}
        if use_python_code:
            desc = f"a[:] = a[-{k}:] + a[:-{k}]"
        else:
            desc = f"Rotate right by {k} positions"
        
    elif operation == "reverse_subrange":
        i = random.randint(0, lst_size - 2)
        j = random.randint(i + 1, lst_size)
        params = {'i': i, 'j': j}
        if use_python_code:
            desc = f"a[{i}:{j}] = reversed(a[{i}:{j}])"
        else:
            desc = f"Reverse subrange from index {i} to {j-1}"
        
    elif operation == "even_odd_indices":
        params = {}
        if use_python_code:
            desc = "a[:] = a[::2] + a[1::2]"
        else:
            desc = "Rearrange to even indices first, then odd indices"
        
    elif operation == "index_select":
        indices = list(range(lst_size))
        random.shuffle(indices)
        params = {'indices': indices}
        if use_python_code:
            indices_str = ', '.join(str(i) for i in indices)
            desc = f"a = [a[{indices_str}]]"
        else:
            desc = f"Reorder using index pattern {indices}"
    
    return desc, params

def generate_list_manipulation_problem(n: int = 5, num_ops: int = None, max_operation_types: int = 7, use_python_code: bool = False) -> tuple[str, List[int]]:
    """
    Generate a list manipulation problem.
    
    Args:
        n: Size of the list
        num_ops: Number of operations to perform. If None, uses random number
        max_operation_types: Maximum number of operation types to use (1-7)
        use_python_code: If True, use Python code strings; if False, natural language
        
    Returns:
        tuple[str, List[int]]: (problem_description, final_list)
    """
    if num_ops is None:
        num_ops = random.randint(max(1, n//2), 2*n)
    
    # Initialize list: [1, 2, 3, 4, 5] for n=5
    initial_list = list(range(1, n+1))
    current_list = initial_list.copy()
    
    # Get available operations based on difficulty limit
    available_operations = generate_list_manipulation_operations(max_operation_types)
    
    # Generate random operations
    operations = []
    for _ in range(num_ops):
        operation = random.choice(available_operations)
        desc, params = generate_operation_with_params(len(current_list), operation, use_python_code)
        operations.append((desc, operation, params))
        
        # Apply operation to track actual result
        current_list = apply_list_operation(current_list, operation, **params)
    
    # Generate problem description with separator tokens
    problem_parts = []
    for i, (desc, _, _) in enumerate(operations):
        problem_parts.append(desc)
        if i < len(operations) - 1:  # Add separator except after last operation
            problem_parts.append("|")
    
    problem_description = " ".join(problem_parts)
    
    return problem_description, current_list

def get_list_manipulation_prompts(b: int = 2) -> tuple[str, str]:
    """
    Get the system prompts for list manipulation worker and manager agents.
    
    Args:
        b: Branching factor for hierarchical processing
        
    Returns:
        tuple[str, str]: (worker_prompt, manager_prompt)
    """
    worker_prompt = """You are a worker agent in a hierarchical system processing ONE Python list manipulation operation with step-by-step verification.

Your task is to apply exactly one Python operation and report the resulting index mapping with perfect accuracy.

You will receive:
- Current list state as index mapping (e.g., [a[2], a[0], a[1]] meaning position 0 has original element 2, position 1 has original element 0, etc.)
- ONE Python operation (e.g., a[0], a[2] = a[2], a[0] or a[::-1])

**CHAIN-OF-THOUGHT PROCESS:**
1. Parse the Python operation syntax carefully
2. Understand the current state mapping
3. Apply the operation step-by-step
4. Verify the result makes sense
5. Output in exact required format

CRITICAL: Use this EXACT template format for your response:

**Example 1:**
Input: Current state: [a[0], a[1], a[2]], Operation: a[0], a[2] = a[2], a[0]
Step 1: Current mapping is [a[0], a[1], a[2]]
Step 2: Python operation a[0], a[2] = a[2], a[0] swaps elements at positions 0 and 2
Step 3: Element at position 0 (a[0]) exchanges with element at position 2 (a[2])
Step 4: New mapping is [a[2], a[1], a[0]]
Verification: Swapped positions 0 and 2, position 1 unchanged ✓
The answer is: [a[2], a[1], a[0]]

**Example 2:**
Input: Current state: [a[1], a[0], a[2]], Operation: a[::-1]
Step 1: Current mapping is [a[1], a[0], a[2]]
Step 2: Python operation a[::-1] reverses the entire list
Step 3: Reverse order: [position 0, position 1, position 2] becomes [position 2, position 1, position 0]
Step 4: New mapping is [a[2], a[0], a[1]]
Verification: Complete reversal applied correctly ✓
The answer is: [a[2], a[0], a[1]]

**Example 3:**
Input: Current state: [a[0], a[1], a[2], a[3], a[4]], Operation: a[:] = a[2:] + a[:2]
Step 1: Current mapping is [a[0], a[1], a[2], a[3], a[4]]
Step 2: Python operation a[:] = a[2:] + a[:2] rotates left by 2 positions
Step 3: a[2:] = [a[2], a[3], a[4]] and a[:2] = [a[0], a[1]]
Step 4: New mapping is [a[2], a[3], a[4], a[0], a[1]]
Verification: Left rotation by 2 positions applied correctly ✓
The answer is: [a[2], a[3], a[4], a[0], a[1]]

**ERROR HANDLING:**
- If operation syntax is unclear, output "PARSE_ERROR: [operation]" and explain issue
- If indices would be out of bounds, output "INDEX_ERROR: [operation]" and skip
- Always verify your result has the same number of elements as input

**PYTHON OPERATION REFERENCE:**
- a[i], a[j] = a[j], a[i] → Swap elements at positions i and j
- a[::-1] → Reverse entire list
- a[:] = a[k:] + a[:k] → Rotate left by k positions  
- a[:] = a[-k:] + a[:-k] → Rotate right by k positions
- a[i:j] = reversed(a[i:j]) → Reverse subrange from i to j-1
- a[:] = a[::2] + a[1::2] → Even indices first, then odd indices

**INSTRUCTIONS:**
- Always follow the 4-step template exactly
- Include verification step to check your work
- Always output in the format [a[i1], a[i2], a[i3], ...] where i1, i2, i3 are the original indices
- Present the final answer in the format "The answer is: [a[i1], a[i2], a[i3], ...]"
- Be precise with Python operation semantics"""

    manager_prompt = f"""You are a manager agent in a hierarchical Python list manipulation system combining results from {b} workers with verification.

Your task is to determine the final index mapping after your workers processed their assigned Python operations in sequence.

**CHAIN-OF-THOUGHT PROCESS:**
1. Verify each worker's result format and validity
2. Understand the sequential processing order
3. Identify the final cumulative state
4. Validate the result makes logical sense
5. Output in exact required format

CRITICAL: Use this EXACT template format for your response:

**Example:**
Input: Worker results: ["[a[1], a[0], a[2]]", "[a[2], a[0], a[1]]"]
Step 1: I have {b} worker results processed sequentially
Step 2: Worker 1 result: [a[1], a[0], a[2]] - valid format ✓
Step 3: Worker 2 result: [a[2], a[0], a[1]] - valid format ✓  
Step 4: The final worker's result [a[2], a[0], a[1]] represents the cumulative effect of all operations
Verification: All workers completed successfully, final state is well-formed ✓
The answer is: [a[2], a[0], a[1]]

**ERROR HANDLING:**
- If any worker result is malformed, output "WORKER_ERROR: [worker_id] - [issue]"
- If worker results are inconsistent, output "CONSISTENCY_ERROR: [explanation]"
- If final result has wrong number of elements, output "SIZE_ERROR: [explanation]"

**VALIDATION CHECKS:**
- All worker results must have valid format [a[i1], a[i2], ...]
- Final result must have same number of elements as expected
- All indices must be valid (0 to n-1 for n elements)
- Sequential processing order must be respected

**INSTRUCTIONS:**
- Workers processed Python operations sequentially (worker 1 → worker 2 → ... → worker {b})
- The LAST worker's index mapping represents the final state after all operations
- Always follow the verification template exactly
- Include validation checks before final answer
- Present the final answer in the format "The answer is: [a[i1], a[i2], a[i3], ...]" exactly as reported by the last worker
- Be precise about which worker provided the final result"""

    return worker_prompt, manager_prompt

def extract_index_mapping(text: str) -> str:
    """
    Extract index mapping from agent response text with enhanced error detection.
    
    Args:
        text: Agent response text containing index mapping
        
    Returns:
        str: Extracted mapping string, or None if not found
    """
    # Check for explicit error messages first
    error_patterns = [
        r"PARSE_ERROR:\s*(.+)",
        r"INDEX_ERROR:\s*(.+)", 
        r"WORKER_ERROR:\s*(.+)",
        r"CONSISTENCY_ERROR:\s*(.+)",
        r"SIZE_ERROR:\s*(.+)"
    ]
    
    for pattern in error_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            # Found an error message, return None to indicate parsing failure
            return None
    
    # Look for primary pattern in the text - match [a[...], a[...], ...]
    match = re.search(r"The answer is:?\s*(\[(?:a\[\d+\](?:,\s*)?)+\])", text, re.IGNORECASE)
    if match:
        result = match.group(1).strip()
        # Validate the extracted pattern
        if _validate_index_mapping_format(result):
            return result
    
    # Fallback: look for any index mapping pattern
    match = re.search(r"\[(?:a\[\d+\](?:,\s*)?)+\]", text)
    if match:
        result = match.group(0).strip()
        if _validate_index_mapping_format(result):
            return result
    
    # Check if response contains partial or malformed patterns
    if re.search(r"a\[\d+\]", text):
        # Contains index notation but not properly formatted
        return None
    
    return None

def _validate_index_mapping_format(mapping_str: str) -> bool:
    """
    Validate that the index mapping string has correct format.
    
    Args:
        mapping_str: String like "[a[2], a[0], a[1]]"
        
    Returns:
        bool: True if format is valid
    """
    try:
        # Check basic structure
        if not (mapping_str.startswith('[') and mapping_str.endswith(']')):
            return False
        
        # Extract all indices
        indices = re.findall(r'a\[(\d+)\]', mapping_str)
        if not indices:
            return False
        
        # Check that indices are valid integers
        for idx_str in indices:
            int(idx_str)  # Will raise ValueError if invalid
        
        return True
    except:
        return False

def parse_index_mapping(mapping_str: str) -> List[int]:
    """
    Parse index mapping string to list of indices.
    
    Args:
        mapping_str: String like "[a[2], a[0], a[1]]"
        
    Returns:
        List[int]: List of original indices [2, 0, 1]
    """
    try:
        # Extract numbers from the pattern [a[i], a[j], a[k], ...]
        indices = re.findall(r'a\[(\d+)\]', mapping_str)
        return [int(idx) for idx in indices]
    except:
        return None

def evaluate_list_manipulation_accuracy(predicted_mapping: List[int], true_list: List[int], original_list: List[int]) -> tuple[bool, float, str]:
    """
    Evaluate list manipulation prediction accuracy with detailed error categorization.
    
    Args:
        predicted_mapping: List of predicted original indices
        true_list: True final list after operations  
        original_list: Original list [1, 2, 3, 4, 5]
        
    Returns:
        tuple[bool, float, str]: (exact_match, element_accuracy, error_type)
    """
    if predicted_mapping is None:
        return False, 0.0, "PARSING_FAILURE"
    
    if len(predicted_mapping) != len(true_list):
        return False, 0.0, "SIZE_MISMATCH"
    
    # Check for invalid indices
    max_index = len(original_list) - 1
    for idx in predicted_mapping:
        if not isinstance(idx, int) or idx < 0 or idx > max_index:
            return False, 0.0, "INVALID_INDICES"
    
    # Convert predicted mapping to actual list
    try:
        predicted_list = [original_list[i] for i in predicted_mapping]
    except (IndexError, TypeError):
        return False, 0.0, "INDEX_ERROR"
    
    # Check exact match
    exact_match = predicted_list == true_list
    
    # Calculate element-wise accuracy
    correct_elements = sum(1 for p, t in zip(predicted_list, true_list) if p == t)
    element_accuracy = correct_elements / len(true_list) if len(true_list) > 0 else 0.0
    
    if exact_match:
        error_type = "NONE"
    elif element_accuracy > 0.0:
        error_type = "PARTIAL_CORRECT"
    else:
        error_type = "COMPLETELY_WRONG"
    
    return exact_match, element_accuracy, error_type

def analyze_error_patterns(error_type_counts: dict, total_runs: int) -> dict:
    """
    Analyze error patterns and calculate error rates.
    
    Args:
        error_type_counts: Dictionary mapping error types to counts
        total_runs: Total number of runs
        
    Returns:
        dict: Analysis of error patterns
    """
    analysis = {
        "total_runs": total_runs,
        "error_rates": {},
        "most_common_error": None,
        "success_rate": 0.0
    }
    
    success_count = error_type_counts.get("NONE", 0)
    analysis["success_rate"] = success_count / total_runs if total_runs > 0 else 0.0
    
    for error_type, count in error_type_counts.items():
        analysis["error_rates"][error_type] = count / total_runs if total_runs > 0 else 0.0
    
    # Find most common error (excluding NONE)
    error_only = {k: v for k, v in error_type_counts.items() if k != "NONE"}
    if error_only:
        analysis["most_common_error"] = max(error_only.items(), key=lambda x: x[1])
    
    return analysis

def validate_intermediate_steps(predicted_mapping: List[int], expected_size: int) -> dict:
    """
    Validate intermediate steps in the prediction process.
    
    Args:
        predicted_mapping: The predicted index mapping
        expected_size: Expected size of the mapping
        
    Returns:
        dict: Validation results
    """
    validation = {
        "size_correct": False,
        "indices_valid": False,
        "has_duplicates": False,
        "missing_indices": [],
        "duplicate_indices": [],
        "out_of_bounds": []
    }
    
    if predicted_mapping is None:
        return validation
    
    # Check size
    validation["size_correct"] = len(predicted_mapping) == expected_size
    
    # Check for valid indices
    expected_indices = set(range(expected_size))
    actual_indices = set(predicted_mapping)
    
    validation["indices_valid"] = all(0 <= idx < expected_size for idx in predicted_mapping)
    validation["missing_indices"] = list(expected_indices - actual_indices)
    validation["duplicate_indices"] = [idx for idx in predicted_mapping if predicted_mapping.count(idx) > 1]
    validation["has_duplicates"] = len(validation["duplicate_indices"]) > 0
    validation["out_of_bounds"] = [idx for idx in predicted_mapping if idx < 0 or idx >= expected_size]
    
    return validation