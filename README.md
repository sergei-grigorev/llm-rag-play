# Gemini RAG

A Rust-based Retrieval-Augmented Generation (RAG) system using Gemini embeddings and Qdrant vector database.

## Features

- Process text files into semantic chunks
- Generate embeddings using Google's Gemini API
- Store and retrieve chunks using Qdrant vector database
- Answer questions based on the content of the document

## Prerequisites

- Rust (with Cargo)
- Qdrant instance (cloud or local)
- Gemini API key

## Setup

1. Clone this repository
2. Copy `.env.example` to `.env` and fill in your API keys and configuration
3. Build the project:
   ```bash
   cargo build --release
   ```

## Usage

```bash
# Process a document and start answering questions
./target/release/gemini-rag /path/to/your/document.txt

# When the app is running, type your questions at the prompt
# Type 'exit' to quit
```

## Environment Variables

- `QDRANT_URL`: URL of your Qdrant instance
- `QDRANT_API_KEY`: API key for Qdrant (if required)
- `GEMINI_API_KEY`: Your Gemini API key
- `GEMINI_EMBEDDINGS_URL`: URL for Gemini embeddings API (optional)
- `GEMINI_GENERATE_URL`: URL for Gemini generation API (optional)
- `RUST_LOG`: Logging level (error, warn, info, debug, trace)

## How it Works

1. The application reads the provided text file
2. The text is split into chunks of approximately 500 tokens each
3. Each chunk is converted to a vector embedding using Gemini
4. Chunks and embeddings are stored in Qdrant
5. When you ask a question, it:
   - Converts your question to an embedding
   - Finds the 4 most similar chunks in the document
   - Uses Gemini to generate an answer based on these chunks
   - Returns the answer to you

## Notes

- If you process the same file again, it will skip the processing and use the existing chunks in Qdrant
- The file is identified by its name, not its content, so if you update the file, you'll need to delete the collection in Qdrant to reprocess it
