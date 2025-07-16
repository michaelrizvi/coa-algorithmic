from chain_of_agents import ChainOfAgents
from chain_of_agents.utils import read_pdf, split_into_chunks
from chain_of_agents.agents import WorkerAgent, ManagerAgent
import os
from dotenv import load_dotenv
import pathlib
import sys
import random

# Set seed
random.seed(42)

# Load environment variables
env_path = pathlib.Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

# Verify API key is loaded
if not os.getenv("TOGETHER_API_KEY"):
    raise ValueError("TOGETHER_API_KEY not found in environment variables")

# Initialize Chain of Agents
parity_worker_prompt = """You are a worker agent responsible for analyzing a portion of a document.
Your task is to provide an analysis of the binary string provided in your chunk and determine if it is even or odd parity."""

parity_manager_prompt = """You are a manager agent responsible for synthesizing information from multiple workers.
Your task is to combine their provided parities and determine the overall parity of the binary string. To compute the aggregate parity, follow these steps:
1. Collect the parity results from all worker agents.
2. Each worker will return either 'even' or 'odd'.
3. Count the number of 'odd' responses.
4. If the count of 'odd' responses is even, the overall parity is 'even'.
5. If the count of 'odd' responses is odd, the overall parity is 'odd'.
6. Return the final parity result.
"""

coa = ChainOfAgents(
    worker_model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    manager_model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    chunk_size=10,  # Reduced chunk size for better handling
    worker_prompt=parity_worker_prompt,
    manager_prompt=parity_manager_prompt
)

# input is a string of random bits to comput the parity on 
input_text = ' '.join(random.choice(['0', '1']) for _ in range(200))
# Compute the parity of the bitstring
if len(input_text) % 2 == 0:
    parity = "even"
else:
    parity = "odd"
query = "What is the parity of the given binary string?"

# Process the text
print("\nProcessing document with Chain of Agents...\n")

chunks = split_into_chunks(input_text, coa.chunk_size, coa.worker_model)
worker_outputs = []
previous_cu = None

print("=" * 80)
print("WORKER RESPONSES")
print("=" * 80 + "\n")

for i, chunk in enumerate(chunks):
    print(f"\n{'='*30} Worker {i+1}/{len(chunks)} {'='*30}")
    worker = WorkerAgent(coa.worker_model, coa.worker_prompt)
    output = worker.process_chunk(chunk, query)
    worker_outputs.append(output)
    print(f"\n{output}\n")

print("\n" + "=" * 80)
print("MANAGER SYNTHESIS")
print("=" * 80 + "\n")

manager = ManagerAgent(coa.manager_model, coa.manager_prompt)
final_output = manager.synthesize(worker_outputs, query)
print(final_output)

print("\n" + "=" * 80)

print("ground truth parity:", parity)