# LangGraph Document Summarizer

A scalable document summarization system built with LangGraph using the Map-Reduce pattern. This system efficiently processes large documents by chunking them, generating individual summaries in parallel, and combining them into a comprehensive final summary.

## Features

- **Map-Reduce Architecture**: Parallel processing of document chunks for efficient summarization
- **Multiple Input Formats**: Supports DOCX, TXT, RTF, ODT, and Markdown files
- **Flexible Input Methods**: Load documents via file path or base64-encoded bytes
- **Configurable Chunking**: Adjustable chunk size and overlap parameters
- **Smart Collapsing**: Automatically collapses summaries when they exceed token limits
- **Async Implementation**: Non-blocking operations suitable for ASGI servers
- **LangGraph Studio Compatible**: Full integration with LangGraph development tools

## Architecture

The summarization graph follows this flow:

```
START → Load File → Chunk File → [Generate Summary (parallel)] →
Collect Summaries → Should Collapse? → Collapse/Final Summary → END
```

### Key Components

1. **Load File**: Reads document content from file path or bytes
2. **Chunk File**: Splits document into manageable chunks using tiktoken
3. **Generate Summary**: Maps each chunk to a summary (runs in parallel)
4. **Collect Summaries**: Aggregates all chunk summaries
5. **Should Collapse**: Checks if summaries need further reduction
6. **Collapse Summaries**: Reduces summary count when needed
7. **Generate Final Summary**: Creates the final consolidated summary

## Installation

### Prerequisites

- Python 3.12 or higher
- OpenAI API key
- LangSmith API key (optional, for tracing)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd langgraph-summarizer
```

2. (Recommeneded) Install [uv](https://docs.astral.sh/uv/) as your python package manager (It's fast!)

3. Create and activate a virtual environment: 
    - If using uv as your package manager:
      ```bash
      uv venv langgraph-summarizer
      source langgraph-summarizer/bin/activate  # On Windows: langgraph-summarizer\Scripts\activate
      ```
    - For regural venv:
      ```bash
      python -m venv langgraph-summarizer
      source langgraph-summarizer/bin/activate  # On Windows: langgraph-summarizer\Scripts\activate
      ```

3. Install dependencies:
    - If using uv:
      ```bash
      uv pip install -r requirements.txt
      ```
    - If using pip:
      ```bash
      pip install -r requirements.txt
      ```

4. Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL_NAME=gpt-4o  # Optional, defaults to gpt-4o
LANGSMITH_API_KEY=your_langsmith_api_key_here  # Optional
LANGCHAIN_TRACING_V2=true  # Optional
LANGSMITH_PROJECT=langgraph-summarizer  # Optional
```

## Usage

### With LangGraph Studio

1. Start LangGraph Studio:

    - If installed dependancies with uv:
      ```bash
      uv run langgraph dev
      ```
    - If installed dependancies with pip:
      ```bash
      langgraph dev
      ```

2. Open the LangGraph Studio UI in your browser (typically `http://localhost:2024`)

3. Send a request with your document:
```json
{
  "file_path": "path/to/your/document.docx",
  "chunk_size": 2000,
  "chunk_overlap": 500
}
```

Or using base64-encoded bytes:
```json
{
  "file_bytes": "base64_encoded_document_content",
  "chunk_size": 2000,
  "chunk_overlap": 500
}
```

### Programmatic Usage

```python
import asyncio
from src.langgraph_summarizer import graph

async def summarize_document():
    result = await graph.ainvoke({
        "file_path": "path/to/document.docx",
        "chunk_size": 2000,
        "chunk_overlap": 500
    })

    print(result["final_summary"])

asyncio.run(summarize_document())
```

### Using the SDK

```python
from langgraph_sdk import get_client

client = get_client(url="http://localhost:2024")

# Create a thread
thread = client.threads.create()

# Run the summarizer
result = client.runs.wait(
    thread['thread_id'],
    "summarizer",
    input={
        "file_path": "document.docx",
        "chunk_size": 2000,
        "chunk_overlap": 500
    }
)

print(result['final_summary'])
```

## Configuration

### Input Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file_path` | str | None | Path to the document file |
| `file_bytes` | str | None | Base64-encoded document bytes |
| `chunk_size` | int | 2000 | Size of each text chunk in tokens |
| `chunk_overlap` | int | 500 | Overlap between consecutive chunks |
| `token_max` | int | 10000 | Maximum tokens before collapsing summaries |

**Note**: Either `file_path` or `file_bytes` must be provided, but not both.

### Supported File Formats

- **DOCX**: Microsoft Word documents
- **TXT**: Plain text files
- **RTF**: Rich Text Format
- **ODT**: OpenDocument Text
- **MD**: Markdown files

## Project Structure

```
langgraph-summarizer/
├── src/
│   ├── langgraph_summarizer.py  # Main graph definition
│   ├── chunker.py               # Text chunking logic
│   ├── file_loader.py           # File loading utilities
│   ├── llm.py                   # LLM configuration
│   ├── logger.py                # Logging setup
│   ├── prompts.py               # System prompts
│   └── utils.py                 # Utility functions
├── test/
│   └── data/                    # Test documents
├── langgraph.json               # LangGraph configuration
├── requirements.txt             # Python dependencies
├── pyproject.toml              # Project metadata
├── .env                        # Environment variables
└── README.md                   # This file
```

## Development

### Testing

Run the test script:
```bash
python test_summarizer.py
```

### Code Quality

The project includes configurations for:
- **Black**: Code formatting (line length: 100)
- **Ruff**: Linting and code quality
- **MyPy**: Static type checking
- **Pytest**: Testing framework

### LangGraph Configuration

The graph is configured in `langgraph.json`:
```json
{
  "dependencies": ["."],
  "graphs": {
    "summarizer": "./src/langgraph_summarizer.py:graph"
  },
  "env": ".env"
}
```

## Key Features Explained

### Map-Reduce Pattern

The system uses LangGraph's `Send` API to implement efficient parallel processing:
1. Document is split into chunks
2. Each chunk is summarized independently (Map phase)
3. Summaries are collected and combined (Reduce phase)
4. If needed, summaries are recursively collapsed until they fit within token limits

### Async Operations

All I/O operations are async to prevent blocking:
- File reading uses `asyncio.to_thread`
- Document processing is non-blocking
- Compatible with ASGI servers

### Error Handling

Robust error handling throughout:
- Invalid file formats are caught early
- Chunk processing errors return descriptive messages
- Failed summaries don't crash the entire pipeline

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure all dependencies are installed via `requirements.txt`

2. **LangGraph blocking errors**: The tiktoken cache is configured to avoid `os.getcwd()` blocking calls

3. **OpenAI API errors**: Verify your API key is valid and has sufficient credits

4. **File encoding issues**: Ensure files are properly encoded (UTF-8 for text files)

### Debug Mode

Enable detailed logging by setting the log level in `src/logger.py`:
```python
logger.setLevel(logging.DEBUG)
```

## Performance Tips

- **Chunk Size**: Larger chunks (10000+) reduce API calls but may lose granularity
- **Chunk Overlap**: 10-15% overlap ensures context continuity
- **Token Max**: Adjust based on your model's context window
- **Parallel Processing**: The system automatically parallelizes chunk processing

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Built with [LangGraph](https://github.com/langchain-ai/langgraph)
- Uses [LangChain](https://github.com/langchain-ai/langchain) for text processing
- Powered by OpenAI's GPT models

## Support

For issues and questions:
- GitHub Issues: [Create an issue](https://github.com/yourusername/langgraph-summarizer/issues)
- Documentation: [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
