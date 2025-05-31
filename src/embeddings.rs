use crate::context::ContextualizedChunk;
use crate::gemini::{Embedding, GeminiClient};
use anyhow::Result;

// Using Embedding from gemini module

// Using GeminiConfig from gemini module

// Using GeminiClient from gemini module

/// Represents an embedding with its associated contextualized chunk
#[derive(Debug, Clone)]
pub struct ContextualEmbedding {
    pub embedding: Embedding,
    pub contextualized_chunk: ContextualizedChunk,
}

// Methods moved to gemini module

/// Extension trait to add contextual embedding methods to GeminiClient
#[allow(async_fn_in_trait)]
pub trait ContextualEmbeddingExt {
    /// Generate embedding for a contextualized chunk
    async fn get_contextual_embedding(
        &self,
        contextualized_chunk: ContextualizedChunk,
    ) -> Result<ContextualEmbedding>;

    /// Generate embeddings for multiple contextualized chunks
    async fn get_contextual_embeddings(
        &self,
        chunks: Vec<ContextualizedChunk>,
    ) -> Result<Vec<ContextualEmbedding>>;
}

impl ContextualEmbeddingExt for GeminiClient {
    /// Generate embedding for a contextualized chunk
    async fn get_contextual_embedding(
        &self,
        contextualized_chunk: ContextualizedChunk,
    ) -> Result<ContextualEmbedding> {
        // Generate embedding for the contextualized text instead of the original chunk
        let embedding = self
            .get_embedding(&contextualized_chunk.contextualized_text)
            .await?;

        Ok(ContextualEmbedding {
            embedding,
            contextualized_chunk,
        })
    }

    /// Generate embeddings for multiple contextualized chunks
    async fn get_contextual_embeddings(
        &self,
        chunks: Vec<ContextualizedChunk>,
    ) -> Result<Vec<ContextualEmbedding>> {
        let mut embeddings = Vec::new();

        for chunk in chunks {
            let embedding = self.get_contextual_embedding(chunk).await?;
            embeddings.push(embedding);
        }

        Ok(embeddings)
    }

    // Using get_embedding from gemini module

    // Using generate_answer from gemini module
}
