use crate::database::QdrantClient;
use crate::embeddings::GeminiClient;
use anyhow::Result;
use std::io::{self, Write};

/// RAG (Retrieval-Augmented Generation) engine
pub struct RagEngine {
    qdrant: QdrantClient,
    gemini: GeminiClient,
}

impl RagEngine {
    /// Create a new RAG engine
    pub fn new(qdrant: QdrantClient, gemini: GeminiClient) -> Self {
        RagEngine { qdrant, gemini }
    }

    /// Check if the collection exists
    pub async fn collection_exists(&self, file_name: &str) -> Result<bool> {
        self.qdrant.collection_exists(file_name).await
    }

    /// Process a file: chunk it, generate embeddings, and store in Qdrant
    pub async fn process_file(&self, content: String, file_name: &str) -> Result<()> {
        // Create a new collection
        self.qdrant.create_collection(file_name).await?;

        // Split content into chunks
        let chunks = crate::chunking::split_into_chunks(&content, file_name);
        println!("Split into {} chunks", chunks.len());

        // Generate embeddings for each chunk
        let mut embeddings = Vec::new();
        for chunk in &chunks {
            let embedding = self.gemini.get_embedding(&chunk.text).await?;
            embeddings.push(embedding);
        }

        // Store chunks in Qdrant
        self.qdrant
            .store_chunks(chunks, embeddings, file_name)
            .await?;

        Ok(())
    }

    /// Run the query loop for a file
    pub async fn run_query_loop(&self, file_name: &str) -> Result<()> {
        println!(
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
                println!("Goodbye!");
                break;
            }

            // Get embedding for the question
            let question_embedding = self.gemini.get_embedding(question).await?;

            // Retrieve relevant chunks
            let chunks = self.qdrant.search(question_embedding, file_name, 4).await?;

            if chunks.is_empty() {
                println!("No relevant information found in the document.");
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

            println!("\n{}", answer);
        }

        Ok(())
    }
}
