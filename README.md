# Chain of Agents Implementation

Chain of Agents (CoA) implementation in Python and Swift - A framework for long-context tasks using Large Language Models (LLMs).

## Overview

This repository implements the Chain-of-Agents framework as described in:

- [Chain of Agents: Large language models collaborating on long-context tasks](https://research.google/blog/chain-of-agents-large-language-models-collaborating-on-long-context-tasks/) (Google Research Blog)
- [Chain of Agents Paper](https://openreview.net/pdf?id=LuCLf4BJsr) (Research Paper)

The Chain of Agents framework enables efficient processing of long-context tasks by:

1. Breaking down large inputs into manageable chunks
2. Using worker agents to process individual chunks
3. Employing a manager agent to synthesize results

## Installation

```bash
git clone https://github.com/rudrankriyam/chain-of-agents.git
cd chain-of-agents
pip install -r requirements.txt
```

### Prerequisites

- Python 3.x
- Xcode 15+ (for macOS app)
- Together API key (sign up at [Together.ai](https://together.ai))


## Usage

### Command Line Version

1. Create a `.env` file in the root directory and add your Together API key:

```bash
echo "TOGETHER_API_KEY=your_api_key_here" >> .env
```

2. Run the example script:

```bash
./run.sh
```

This will:
- Set up a Python virtual environment
- Install required dependencies
- Process a sample PDF document using the Chain of Agents framework

### macOS App

1. Start the API server:

```bash
./run_api.sh
```
This will:
- Activate the Python virtual environment
- Start the API server that the macOS app communicates with
- The server will run on `localhost:8000`

2. Open the Xcode project:

```bash
open ChainOfAgents.xcodeproj
```

3. Build and run the app (⌘R)

The macOS app provides:
- PDF document selection
- Custom query input
- Real-time processing visualization
- Worker agent progress tracking
- Final synthesis display

Note: The macOS app requires the API server to be running, so make sure to start it with `run_api.sh` before launching the app.

## Features

- Support for PDF document analysis
- Configurable chunk sizes for processing
- Real-time progress tracking
- Streaming responses from both worker and manager agents
- Clean macOS native interface
- Support for Together AI's LLaMA models

## Models

The project uses Together AI's hosted models:

```python 
- Worker model: `meta-llama/Llama-3.3-70B-Instruct-Turbo-Free`
- Manager model: `meta-llama/Llama-3.3-70B-Instruct-Turbo-Free`
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

```bibtex
@article{zhang2024chain,
    title={Chain of Agents: Large Language Models Collaborating on Long-Context Tasks},
    author={Zhang, Yusen and Sun, Ruoxi and Chen, Yanfei and Pfister, Tomas and Zhang, Rui and Arık, Sercan Ö.},
    journal={arXiv preprint arXiv:2404.08392},
    year={2024}
}
```