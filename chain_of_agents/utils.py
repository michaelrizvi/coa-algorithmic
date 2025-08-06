from typing import List
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
    """
    manager_prompt = f"""You are a manager agent responsible for synthesizing the results of previous workers.
    Your task is to return the parity of the binary string provided by the worker agents. You may think step by step, but your final answer should be concise and clear.
    To compute the parity, follow these steps:
    1. Collect the results from the worker agents. This should be a list of binary digits (0 or 1).
    2. If the parity of the list is even, return 0.
    3. If the parity of the list is odd, return 1.
    5. Present the final answer on a new line in the format "The answer is: [your answer]" 
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
    """

    parity_manager_prompt = """You are a manager agent responsible for synthesizing information from multiple workers.
    Your task is to combine their provided parities and determine the overall parity of the binary string. To compute the aggregate parity, follow these steps:
    1. Collect the parity results from all worker agents.
    2. Each worker will return either 0 or 1.
    3. Count the number of 1 responses.
    4. If the count of 1 responses is even, the overall parity is 0.
    5. If the count of 1 responses is odd, the overall parity is 1.
    6. Present the final answer in the format "The answer is: [your answer]"
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