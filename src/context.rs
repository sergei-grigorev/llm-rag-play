use crate::chunking::TextChunk;
use crate::embeddings::GeminiClient;
use anyhow::Result;

/// Represents a text chunk with added contextual information
#[derive(Debug, Clone)]
pub struct ContextualizedChunk {
    /// The original chunk
    pub original_chunk: TextChunk,
    /// The chunk with prepended context
    pub contextualized_text: String,
    /// Token count of the contextualized text
    pub token_count: usize,
}

/// Context Generator for enhancing chunks with document context
pub struct ContextGenerator {
    gemini_client: GeminiClient,
}

impl ContextGenerator {
    /// Create a new context generator
    pub fn new(gemini_client: GeminiClient) -> Self {
        ContextGenerator { gemini_client }
    }

    /// Generate contextual information for a chunk
    pub async fn generate_context_for_chunk(
        &self,
        chunk: TextChunk,
        source_document: &str,
    ) -> Result<ContextualizedChunk> {
        // If the source document is empty, use a minimal context
        if source_document.is_empty() {
            let contextualized_text = format!(
                "Context: This is a standalone text excerpt.\n\n{}",
                chunk.text
            );
            let token_count = crate::chunking::estimate_token_count(&contextualized_text);

            return Ok(ContextualizedChunk {
                original_chunk: chunk,
                contextualized_text,
                token_count,
            });
        }

        // Create the prompt for context generation
        let prompt = format!(
            "<document>\n{}\n</document>\nHere is the chunk we want to situate within the whole document\n<chunk>\n{}\n</chunk>\nPlease give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else.",
            source_document,
            chunk.text
        );

        // Generate context using Gemini
        let context = self.gemini_client.generate_answer(&prompt, "").await?;

        // Combine the generated context with the original chunk
        let contextualized_text = format!("Context: {}\n\n{}", context.trim(), chunk.text);
        let token_count = crate::chunking::estimate_token_count(&contextualized_text);

        Ok(ContextualizedChunk {
            original_chunk: chunk,
            contextualized_text,
            token_count,
        })
    }

    /// Process a batch of chunks to add context
    pub async fn contextualize_chunks(
        &self,
        chunks: Vec<TextChunk>,
        source_document: &str,
    ) -> Result<Vec<ContextualizedChunk>> {
        let mut contextualized_chunks = Vec::new();

        for chunk in chunks {
            let contextualized_chunk = self
                .generate_context_for_chunk(chunk, source_document)
                .await?;
            contextualized_chunks.push(contextualized_chunk);
        }

        Ok(contextualized_chunks)
    }
}
