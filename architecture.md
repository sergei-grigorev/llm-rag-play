# Gemini RAG Architecture

This document provides a detailed explanation of the Gemini RAG system architecture, components, and algorithms. It serves as a comprehensive reference for understanding how the system processes documents, generates embeddings, and retrieves relevant information.

## System Overview

Gemini RAG is a Rust-based Retrieval-Augmented Generation system that leverages Google's Gemini API for embeddings and text generation, with Qdrant as the vector database for efficient semantic search. The system implements advanced RAG patterns, including contextual retrieval for improved accuracy and memory-optimized document handling.

## Core Components

The system is organized into several modular components, each responsible for a specific aspect of the RAG pipeline:

### 1. Document Module (`document.rs`)

Handles document processing and type detection:

- **Document Structure**:
  - `document_id`: Unique identifier for the document
  - `content`: The document's text content
  - `mime_type`: Detected MIME type (e.g., text/plain, application/pdf)
  - `metadata`: Additional document metadata

- **Document Processing**:
  - Automatically detects document type using MIME type checking
  - Handles both text and PDF files
  - Normalizes whitespace and cleans up text
  - Implements efficient memory management with references

### 2. Chunking Module (`chunking.rs`)

Responsible for splitting documents into manageable chunks for processing:

- **TextChunk Structure**: Represents a text chunk with metadata including:
  - `text`: The actual content of the chunk
  - `token_count`: Estimated token count for the chunk
  - `document_id`: Identifier for the source document
  - `start_position`: Position in the original document

- **Chunking Algorithm**:
  - Splits documents into ~500 token chunks with 50-token overlap
  - Uses a hierarchical approach: first splitting by paragraphs, then by sentences if needed
  - Handles large paragraphs by breaking them into smaller units
  - Maintains overlap between chunks to preserve context across boundaries
  - Recursively processes chunks that exceed size limits

- **Memory Optimization**:
  - Stores document references (ID and position) instead of duplicating the entire document
  - Each chunk maintains metadata that can be used to locate the original content

### 3. Context Module (`context.rs`)

Implements contextual retrieval to enhance chunks with additional information:

- **ContextualizedChunk Structure**:
  - Contains the original `TextChunk`
  - Stores the contextualized text (original text with prepended context)
  - Tracks token count of the enhanced text

- **Context Generation Process**:
  - For each chunk, generates a succinct description of its position and role in the document
  - Uses the Gemini API to create context by analyzing the relationship between the chunk and the full document
  - Prepends the generated context to the chunk text with a "Context:" prefix
  - Maintains references to the original chunks to avoid data duplication

### 4. Embeddings Module (`embeddings.rs`)

Handles interaction with the Gemini API for generating embeddings and text:

- **GeminiClient**:
  - Manages API authentication and requests
  - Provides methods for generating embeddings and text responses

- **Embedding Generation**:
  - Converts contextualized chunks to vector embeddings using Gemini's embeddings-004 model
  - Processes both individual chunks and batches of chunks
  - Returns embeddings along with their associated contextualized chunks

- **Answer Generation**:
  - Takes retrieved context and user questions as input
  - Formats prompts to include both context and question
  - Configures generation parameters (temperature, top_p, top_k, max_output_tokens)
  - Returns generated answers based on the provided context

### 5. Database Module (`database.rs`)

Manages vector storage and retrieval using Qdrant:

- **QdrantClient**:
  - Handles connection to Qdrant instance
  - Provides methods for collection management and vector operations

- **Collection Management**:
  - Creates collections with appropriate vector parameters (dimension, distance metric)
  - Checks for collection existence to avoid reprocessing
  - Handles collection naming based on document identifiers

- **Vector Operations**:
  - Stores chunks and their embeddings as points in Qdrant
  - Preserves metadata in the payload (text, document_id, start_position)
  - Performs semantic search using cosine similarity
  - Retrieves and reconstructs TextChunks from search results

### 6. RAG Engine (`rag.rs`)

Orchestrates the entire RAG workflow:

- **Process Flow**:
  - Initializes necessary components (Qdrant client, Gemini client, Context generator)
  - Coordinates document processing, chunking, context generation, and embedding
  - Manages the query loop for interactive Q&A

- **Document Processing Pipeline**:
  1. Reads and chunks the input document
  2. Generates contextual information for each chunk
  3. Creates embeddings for contextualized chunks
  4. Stores chunks and embeddings in Qdrant

- **Query Processing Pipeline**:
  1. Converts user questions to embeddings
  2. Retrieves relevant chunks using vector similarity search
  3. Combines retrieved chunks to form a comprehensive context
  4. Generates answers based on the context and question

## Enhanced Features

### Multi-Model Support

The system now supports multiple Gemini models for different tasks:

- **Gemini 2.5 Flash Preview 05-20**: Used for question answering
- **Gemini 2.0 Flash-Lite**: Optimized for context generation
- Each model is selected based on the specific task requirements

### Progress Tracking

Enhanced processing feedback includes:

- Real-time progress updates during document processing
- Percentage completion indicators
- Chunk processing statistics
- Error reporting with detailed messages

## Algorithmic Details

### Chunking Algorithm

The system uses a sophisticated chunking algorithm that:

1. Splits text by paragraphs first (on double newlines)
2. For each paragraph:
   - If the paragraph is smaller than the target size (500 tokens), add it to the current chunk
   - If adding would exceed the target size, finalize the current chunk and start a new one with overlap
   - If a single paragraph is too large, split it into sentences and process each sentence
3. Ensures chunks have appropriate overlap (50 tokens) to maintain context
4. Handles edge cases like extremely large paragraphs through recursive processing

### Contextual Retrieval Implementation

The contextual retrieval approach enhances standard RAG by:

1. **Context Generation**:
   - For each chunk, analyzes its relationship to the full document
   - Generates a concise description of the chunk's position and content
   - Uses the Gemini model to create this context

2. **Contextualized Embeddings**:
   - Prepends the generated context to each chunk
   - Generates embeddings for these contextualized chunks
   - Stores both the contextualized text and metadata

3. **Memory-Efficient Storage**:
   - Maintains references to original document positions
   - Avoids storing duplicate copies of source documents
   - Uses efficient data structures to minimize memory footprint

### Vector Search and Retrieval

The search process involves:

1. Converting the user query to an embedding using the same model
2. Performing vector similarity search in Qdrant using cosine distance
3. Retrieving the most semantically similar chunks (default: top 4)
4. Reconstructing TextChunks from the search results
5. Combining chunks to form a comprehensive context for answer generation

## Memory Optimization

The system implements advanced memory optimization through:

1. **Reference-Based Architecture**:
   - Stores document references instead of full copies
   - Uses efficient string interning for repeated terms
   - Implements lazy loading for large documents

2. **Efficient Data Structures**:
   - Uses Rust's ownership model to prevent unnecessary copies
   - Implements custom allocators for chunk storage
   - Uses memory mapping for large files

3. **Document Processing**:
   - Processes documents in streaming fashion
   - Implements chunk recycling for large documents
   - Uses zero-copy parsing where possible

1. **Reference-Based Storage**:
   - TextChunk structure stores metadata (document ID, position) instead of duplicating the entire document
   - Each chunk maintains references to its position in the source document

2. **Efficient Data Structures**:
   - Uses Rust's ownership model to avoid unnecessary copying
   - Implements clone operations only when necessary
   - Maintains references to shared data where possible

3. **Document Reference Handling**:
   - Enables tracking of content origin without redundant storage
   - Allows reconstruction of original context when needed

## Performance Considerations

### Processing Speed

- Documents are processed in parallel when possible
- Asynchronous I/O for network operations
- Batch processing of embeddings
- Connection pooling for database access

### Resource Management

- Configurable memory limits
- Automatic cleanup of temporary files
- Graceful handling of out-of-memory conditions
- Efficient error recovery mechanisms

## Workflow Sequence

### Document Ingestion and Processing

1. User provides a document path
2. System checks if the document has already been processed
3. If not processed:
   - Document is read and split into chunks
   - Context is generated for each chunk
   - Contextualized chunks are converted to embeddings
   - Chunks and embeddings are stored in Qdrant

### Question Answering

1. User enters a question
2. Question is converted to an embedding
3. System searches for similar chunks in Qdrant
4. Retrieved chunks are combined to form context
5. Gemini generates an answer based on the context and question
6. Answer is presented to the user

## Configuration and Extensibility

The system is designed to be configurable through:

1. **Environment Variables**:
   - API keys and endpoints
   - Database connection details
   - Logging levels

2. **Extensible Components**:
   - The Context Generator can be extended for domain-specific enrichment
   - Chunking parameters can be adjusted for different document types
   - Vector search parameters can be tuned for precision vs. recall

## Performance Considerations

For optimal performance, the system:

1. Uses efficient chunking to balance context preservation and processing speed
2. Implements batch processing where possible
3. Avoids redundant API calls and database operations
4. Uses asynchronous processing with Tokio for I/O operations
5. Minimizes memory usage through reference-based document handling

## Future Enhancement Opportunities

The architecture supports several potential enhancements:

1. **BM25 Hybrid Search**: Combining vector search with keyword-based retrieval
2. **Reranking**: Adding a post-retrieval ranking step to improve relevance
3. **Streaming Responses**: Implementing streaming for real-time answer generation
4. **Multi-Document Support**: Extending to handle multiple documents in a single session
5. **Incremental Updates**: Supporting document updates without full reprocessing
