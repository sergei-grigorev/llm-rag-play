# Gemini RAG

A high-performance Rust-based Retrieval-Augmented Generation (RAG) system leveraging Google's Gemini API for embeddings and text generation, with Qdrant vector database for efficient semantic search. The system implements advanced RAG patterns including contextual retrieval for improved accuracy and memory-optimized document handling.

## Features

- Process text files into optimized semantic chunks with configurable overlap
- Generate embeddings using Google's Gemini API
- Efficient vector storage and retrieval with Qdrant
- Contextual retrieval for improved answer quality
- Memory-optimized architecture with document reference handling
- Support for vector similarity search
- Configurable chunking and retrieval parameters

## Prerequisites

- Rust (with Cargo)
- Qdrant instance (cloud or local)
- Gemini API key
- [just](https://github.com/casey/just) (for development tasks, optional)

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

## Development

This project uses [just](https://github.com/casey/just) for running common development tasks. Install it with:

```bash
cargo install just
```

Available commands:

- `just` - Show available commands
- `just format` - Format code using rustfmt
- `just lint` - Lint code using clippy
- `just build` - Build the project in release mode
- `just test` - Run tests
- `just run` - Run the application
- `just clean` - Clean build artifacts

## Environment Variables

- `QDRANT_URL`: URL of your Qdrant instance
- `QDRANT_API_KEY`: API key for Qdrant (if required)
- `GEMINI_API_KEY`: Your Gemini API key
- `GEMINI_EMBEDDINGS_URL`: URL for Gemini embeddings API (optional)
- `GEMINI_GENERATE_URL`: URL for Gemini generation API (optional)
- `RUST_LOG`: Logging level (error, warn, info, debug, trace)

## How it Works

1. **Document Processing**
   - Input text is split into chunks of approximately 500 tokens with 50-token overlap
   - Each chunk maintains metadata including document ID and position in the source document
   - Memory-optimized storage avoids storing duplicate copies of source documents

2. **Embedding Generation**
   - Each chunk is enhanced with contextual information about its position and content
   - Contextualized chunks are converted to vector embeddings using Gemini's embeddings-004 model
   - The system stores the contextualized text along with metadata referencing the original document

3. **Vector Storage**
   - Contextualized chunks and their embeddings are stored in Qdrant with efficient indexing
   - Each vector point contains payload with the chunk text and metadata (document ID, position)
   - Supports cosine similarity search for semantic retrieval

4. **Question Answering**
   - User questions are converted to embeddings using the same Gemini model
   - The system retrieves the most relevant chunks using vector similarity search
   - Retrieved chunks are combined to form a comprehensive context
   - Gemini generates accurate, source-grounded answers based on the retrieved context

5. **Memory Optimization**
   - TextChunk structure stores metadata (document ID, position) instead of duplicating the entire document
   - Efficient data structures minimize memory footprint
   - Document references enable tracking of content origin without redundant storage

## Notes

- The system uses file hashing to detect changes - reprocessing only occurs when content changes
- Document collections are versioned to support updates without data loss
- The context module can be extended to support domain-specific enrichment
- For large document sets, consider adjusting chunk size and overlap for optimal performance

## Recent Improvements

- **Memory Optimization**: Reduced memory usage through reference-based document storage
- **Contextual Retrieval**: Improved answer quality with contextual embeddings
- **Context Generation**: Added automatic context generation for chunks to improve retrieval
- **Enhanced Documentation**: Added comprehensive [architecture documentation](./architecture.md) and usage examples

### Document Processing Flow

1. **Document Ingestion**: The system reads the input document.
2. **Chunking**: The document is split into chunks of ~500 tokens with 50-token overlap.
3. **Context Generation**: Each chunk is enhanced with contextual information about its position and content within the document.
4. **Embedding Generation**: The contextualized chunks are converted to vector embeddings using Gemini's embeddings-004 model.
5. **Vector Storage**: The embeddings and chunks are stored in Qdrant with efficient reference handling to minimize memory usage.

### Query Processing Flow

1. **Query Analysis**: The user's question is processed.
2. **Query Embedding**: The question is converted to a vector embedding using the same model.
3. **Vector Search**: The system searches for the most semantically similar chunks in the vector database.
4. **Chunk Retrieval**: The most relevant chunks are retrieved and combined.
5. **Answer Generation**: The Gemini model generates an answer based on the retrieved context and the original question.
