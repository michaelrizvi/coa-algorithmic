from typing import List, Optional, Iterator, Dict
from together import Together
import os

class WorkerAgent:
    """Worker agent that processes individual chunks of text."""
    
    def __init__(self, model: str, system_prompt: str, max_tokens: int = 512):
        """
        Initialize a worker agent.
        
        Args:
            model: The LLM model to use (e.g., "gpt-3.5-turbo")
            system_prompt: The system prompt that defines the worker's role
        """
        self.model = model
        self.system_prompt = system_prompt
        self.client = Together()  # Uses TOGETHER_API_KEY from environment
        self.max_tokens = max_tokens  # Adjust as needed for your model
    
    def process_chunk(self, chunk: str, query: str, previous_cu: Optional[str] = None) -> str:
        """
        Process a single chunk of text.
        
        Args:
            chunk: The text chunk to process
            query: The user's query
            previous_cu: The previous Cognitive Unit (CU) if any
            
        Returns:
            str: The processed output for this chunk
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Chunk: {chunk}\nQuery: {query}\nPrevious CU: {previous_cu or 'None'}"}
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
            max_tokens=self.max_tokens
        )
        
        return response.choices[0].message.content
    
    async def process_chunk_stream(self, chunk: str, query: str, previous_cu: Optional[str] = None) -> Iterator[str]:
        """Process a chunk with streaming (for future implementation)."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Chunk: {chunk}\nQuery: {query}\nPrevious CU: {previous_cu or 'None'}"}
        ]
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
            stream=True
        )
        
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

class ManagerAgent:
    """Manager agent that synthesizes outputs from worker agents."""
    
    def __init__(self, model: str, system_prompt: str, max_tokens: int = 1024):
        """
        Initialize a manager agent.
        
        Args:
            model: The LLM model to use (e.g., "gpt-4")
            system_prompt: The system prompt that defines the manager's role
        """
        self.model = model
        self.system_prompt = system_prompt
        self.client = Together()  # Uses TOGETHER_API_KEY from environment
        self.max_new_tokens = max_tokens  # Adjust as needed for synthesis
    
    def synthesize(self, worker_outputs: List[str], query: str) -> str:
        """
        Synthesize outputs from multiple worker agents.
        
        Args:
            worker_outputs: List of outputs from worker agents
            query: The original user query
            
        Returns:
            str: The final synthesized response
        """
        combined_outputs = "\n\n".join(f"Worker {i+1}: {output}" 
                                     for i, output in enumerate(worker_outputs))
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Worker Outputs:\n{combined_outputs}\n\nQuery: {query}"}
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
            max_new_tokens=self.max_new_tokens
        )
        
        return response.choices[0].message.content
    
    async def synthesize_stream(self, worker_outputs: List[str], query: str) -> Iterator[str]:
        """Synthesize with streaming (for future implementation)."""
        combined_outputs = "\n\n".join(f"Worker {i+1}: {output}" 
                                     for i, output in enumerate(worker_outputs))
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Worker Outputs:\n{combined_outputs}\n\nQuery: {query}"}
        ]
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
            stream=True
        )
        
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content 