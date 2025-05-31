use crate::context::ContextGenerator;
use crate::database::QdrantClient;
use crate::embeddings::GeminiClient;
use anyhow::Result;
use log::info;
use std::io::{self, Write};

/// RAG (Retrieval-Augmented Generation) engine
pub struct RagEngine {
    qdrant: QdrantClient,
    gemini: GeminiClient,
    context_generator: ContextGenerator,
}

impl RagEngine {
    /// Create a new RAG engine
    pub fn new(qdrant: QdrantClient, gemini: GeminiClient) -> Self {
        // Create a context generator using the same Gemini client
        let context_generator = ContextGenerator::new(gemini.clone());

        RagEngine {
            qdrant,
            gemini,
            context_generator,
        }
    }

    /// Check if the collection exists
    pub async fn collection_exists(&self, file_name: &str) -> Result<bool> {
        self.qdrant.collection_exists(file_name).await
    }

    /// Process a file: chunk it, generate embeddings, and store in Qdrant
    pub async fn process_file(&self, content: String, file_name: &str) -> Result<()> {
        // We need to ensure the content string lives long enough
        let content_ref = &content;
        // Create a new collection
        self.qdrant.create_collection(file_name).await?;

        // Split content into chunks
        let chunks = crate::chunking::split_into_chunks(content_ref, file_name);
        info!("Split into {} chunks", chunks.len());

        // Generate context for each chunk
        info!("Generating contextual information for chunks...");
        let contextualized_chunks = self
            .context_generator
            .contextualize_chunks(chunks, &content)
            .await?;
        info!(
            "Generated context for {} chunks",
            contextualized_chunks.len()
        );

        // Generate embeddings for contextualized chunks
        info!("Generating embeddings for contextualized chunks...");
        let contextual_embeddings = self
            .gemini
            .get_contextual_embeddings(contextualized_chunks)
            .await?;

        // Create new chunks with contextualized text but preserve metadata
        let mut contextualized_chunks_for_storage = Vec::new();
        let mut embeddings = Vec::new();

        for contextual_embedding in contextual_embeddings {
            // Create a new TextChunk with contextualized text but same metadata
            let original_chunk = contextual_embedding.contextualized_chunk.original_chunk;
            let contextualized_text_chunk = crate::chunking::TextChunk {
                text: contextual_embedding
                    .contextualized_chunk
                    .contextualized_text,
                token_count: contextual_embedding.contextualized_chunk.token_count,
                document_id: original_chunk.document_id,
                start_position: original_chunk.start_position,
            };

            contextualized_chunks_for_storage.push(contextualized_text_chunk);
            embeddings.push(contextual_embedding.embedding);
        }

        // Store contextualized chunks in Qdrant
        self.qdrant
            .store_chunks(contextualized_chunks_for_storage, embeddings, file_name)
            .await?;

        Ok(())
    }

    /// Run the query loop for a file
    pub async fn run_query_loop(&self, file_name: &str) -> Result<()> {
        info!(
            "Ready to answer questions about {}. Type 'exit' to quit.",
            file_name
        );

        let stdin = io::stdin();
        let mut stdout = io::stdout();
        let mut buffer = String::new();

        loop {
            print!("\nYour question: ");
            stdout.flush()?;

            buffer.clear();
            stdin.read_line(&mut buffer)?;

            let question = buffer.trim();

            if question.to_lowercase() == "exit" {
                info!("Goodbye!");
                break;
            }

            // Get embedding for the question
            let question_embedding = self.gemini.get_embedding(question).await?;

            // Retrieve relevant chunks
            let chunks = self.qdrant.search(question_embedding, file_name, 4).await?;

            if chunks.is_empty() {
                info!("No relevant information found in the document.");
                continue;
            }

            // Create context from chunks
            let context = chunks
                .iter()
                .map(|chunk| chunk.text.clone())
                .collect::<Vec<String>>()
                .join("\n\n");

            // Generate answer
            let answer = self.gemini.generate_answer(&context, question).await?;

            info!("\n{}", answer);
        }

        Ok(())
    }
}
