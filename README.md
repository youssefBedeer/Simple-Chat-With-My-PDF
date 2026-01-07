# RAG Agent — Chat with a PDF

This repository provides a simple Retrieval-Augmented Generation (RAG) agent that lets you chat with a PDF document. The agent embeds PDF text into a Chroma vector store, uses a retriever to find relevant chunks, and queries an LLM to answer user questions while citing document excerpts.

## Files
- [RAG.py](RAG.py): Main script — loads a PDF, builds a vector store, and runs an interactive chat loop.
- `db/`: Persistence directory used by Chroma (created when the vector DB is persisted).

## Features
- Load a PDF and split it into chunks using `RecursiveCharacterTextSplitter`.
- Embed chunks using local Hugging Face embeddings.
- Store embeddings in a local Chroma vector DB and retrieve the top-k results.
- Use an LLM (via OpenRouter/ChatOpenAI in the script) with a tool binding to call the retriever.
- Interactive REPL for asking questions about the PDF and getting cited answers.

## Requirements
- Python 3.10+ recommended
- Install dependencies (example):

```bash
pip install langchain langchain-chroma langchain-openai langchain-community langgraph python-dotenv
```

Notes:
- Package names may vary depending on distribution. If installation fails, consult the package documentation for the `langchain` community integrations you use.

## Environment variables
Create a `.env` file in the same folder as [RAG.py](RAG.py) with at least:

```text
OPENROUTER_API_KEY=your_openrouter_api_key_here
# Optionally: path to a local Hugging Face embeddings model
# HUGGINGFACE_MODEL_PATH=C:\path\to\huggingface_model_folder
```

The script uses `OPENROUTER_API_KEY` to call the ChatOpenAI model via OpenRouter. If you use a different LLM backend, update the `llm` initialization in `RAG.py` accordingly.

## Configuration (edit `RAG.py`)
- `pdf_path`: Set this to the path of your PDF. Example in the script:

```python
pdf_path = r"D:\Programming\ML\Agentic Ai course\FreeCodeCamp course\Stock_Market_Performance_2024.pdf"
```

- `persist_directory`: Directory where Chroma persistence files are stored (defaults to `db/`).
- `embeddings` configuration: The script uses a local Hugging Face embeddings model path; update if you want a different model or use an online embedding provider.

## Running the script
1. Ensure the `.env` file is present and the required packages are installed.
2. Update `pdf_path` in [RAG.py](RAG.py) or pass a path via modifying the script.
3. Run the script:

```bash
python RAG.py
```

You will enter an interactive prompt. Type questions about the PDF; type `exit` or `quit` to stop.

Example:

```
Your question: What were the key drivers of stock market performance in 2024?

=== ANSWER ===
<LLM answer citing document excerpts>
```

## How it works (high level)
1. Load PDF pages with `PyPDFLoader`.
2. Split pages into overlapping chunks using `RecursiveCharacterTextSplitter`.
3. Create embeddings for each chunk with `HuggingFaceEmbeddings`.
4. Persist embeddings to Chroma and expose a retriever.
5. Bind a retriever tool to the LLM so the LLM can call the tool to fetch relevant document text.
6. A state graph loops between the LLM and the retriever until no tool calls remain, then returns the final answer.

## Troubleshooting
- PDF not found: Ensure `pdf_path` points to the correct file and has proper read permissions.
- Embeddings load errors: Verify the `HUGGINGFACE_MODEL_PATH` (or model name) and that the model files exist and are compatible with `HuggingFaceEmbeddings`.
- LLM/auth errors: Verify `OPENROUTER_API_KEY` and network access to the `base_url` in the `ChatOpenAI` initialization.
- Chroma DB issues: If persistence fails, remove or rename the `db/` folder and let the script recreate it.

