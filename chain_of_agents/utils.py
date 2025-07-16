from typing import List
import logging
import fitz  # PyMuPDF
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def get_majority_vote_prompt() -> str:
    """
    Get the system prompt for majority vote synthesis.
    
    Returns:
        str: The majority vote synthesis prompt
    """
    return """You are a reasoning agent tasked with finding the best answer to a problem.

    Given the user's query and the input text, you need to:
    1. Carefully read and understand the problem being asked
    2. Analyze the provided information systematically
    3. Think through the solution step-by-step
    4. Show your reasoning process clearly
    5. Present the final answer in the format "The answer is: [your answer]"

    Your response must be consice, logical, and directly address the query.
    """

def extract_answer(text):
    match = re.search(r"The answer is:?\s*(.+?)(?:\\|\n|$)", text, re.IGNORECASE)
    return match.group(1).strip() if match else None
