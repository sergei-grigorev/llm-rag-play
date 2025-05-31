use crate::chunking::{estimate_token_count, TextChunk};
use crate::gemini::GeminiClient;
use anyhow::Result;
use log::{info, warn};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::time::sleep;

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
    rate_limiter: Arc<Mutex<RateLimiter>>,
}

impl ContextGenerator {
    /// Create a new context generator
    pub fn new(gemini_client: GeminiClient) -> Self {
        ContextGenerator {
            gemini_client,
            rate_limiter: Arc::new(Mutex::new(RateLimiter::new(30, 1_000_000))),
        }
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
            let token_count = estimate_token_count(&contextualized_text);

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

        // Create a custom request for context generation using Gemini 2.0 Flash-Lite
        let context = self.generate_context_with_flash_lite(&prompt).await?;

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

        // Get total number of chunks for progress reporting
        let total_chunks = chunks.len();
        info!("Contextualizing {} chunks...", total_chunks);

        for (i, chunk) in chunks.into_iter().enumerate() {
            let contextualized_chunk = self
                .generate_context_for_chunk(chunk, source_document)
                .await?;
            contextualized_chunks.push(contextualized_chunk);

            // Log progress after every 5th chunk
            if (i + 1) % 5 == 0 {
                info!(
                    "Context progress: processed {}/{} chunks ({}%)",
                    i + 1,
                    total_chunks,
                    ((i + 1) * 100) / total_chunks
                );
            }
        }

        // Log completion if total chunks is not a multiple of 5
        if total_chunks % 5 != 0 {
            info!(
                "Context progress: completed all {}/{} chunks (100%)",
                total_chunks, total_chunks
            );
        }

        Ok(contextualized_chunks)
    }

    /// Generate context using Gemini 2.0 Flash-Lite model specifically for summarization
    /// Rate limited to 30 RPM and 1,000,000 TPM for prompts
    async fn generate_context_with_flash_lite(&self, prompt: &str) -> Result<String> {
        // Estimate token count for the prompt
        let prompt_token_count = estimate_token_count(prompt);

        // Apply rate limiting
        let wait_duration = {
            let mut limiter = self.rate_limiter.lock().unwrap();
            limiter.check_and_update(prompt_token_count)
        };

        if !wait_duration.is_zero() {
            warn!(
                "Rate limit reached, waiting for {:?} before sending request",
                wait_duration
            );
            sleep(wait_duration).await;
        }

        // Use the gemini module's generate_context method
        self.gemini_client.generate_context(prompt).await
    }
}

/// Rate limiter for API requests
struct RateLimiter {
    /// Maximum requests per minute
    max_rpm: usize,
    /// Maximum tokens per minute for prompts
    max_tpm: usize,
    /// Timestamps of recent requests
    request_timestamps: Vec<Instant>,
    /// Token counts of recent requests
    token_counts: Vec<usize>,
}

impl RateLimiter {
    /// Create a new rate limiter
    fn new(max_rpm: usize, max_tpm: usize) -> Self {
        RateLimiter {
            max_rpm,
            max_tpm,
            request_timestamps: Vec::new(),
            token_counts: Vec::new(),
        }
    }

    /// Check if the rate limit has been reached and update the internal state
    /// Returns the duration to wait if the rate limit has been reached
    fn check_and_update(&mut self, token_count: usize) -> Duration {
        let now = Instant::now();
        let one_minute_ago = now - Duration::from_secs(60);

        // Remove entries older than 1 minute
        let mut i = 0;
        while i < self.request_timestamps.len() {
            if self.request_timestamps[i] < one_minute_ago {
                self.request_timestamps.remove(i);
                self.token_counts.remove(i);
            } else {
                i += 1;
            }
        }

        // Calculate current rates
        let current_rpm = self.request_timestamps.len();
        let current_tpm: usize = self.token_counts.iter().sum();

        // Check if adding this request would exceed limits
        if current_rpm >= self.max_rpm || current_tpm + token_count > self.max_tpm {
            // Log which limit was exceeded
            if current_rpm >= self.max_rpm {
                warn!(
                    "Rate limit exceeded: {}/{} requests per minute",
                    current_rpm, self.max_rpm
                );
            }
            if current_tpm + token_count > self.max_tpm {
                warn!(
                    "Token limit exceeded: {}/{} tokens per minute (trying to add {} tokens)",
                    current_tpm, self.max_tpm, token_count
                );
            }

            // Calculate how long to wait
            if !self.request_timestamps.is_empty() {
                let oldest_timestamp = self.request_timestamps[0];
                // Calculate when the oldest request will expire from the window
                let expiry_time = oldest_timestamp + Duration::from_secs(60);
                let wait_duration = if expiry_time > now {
                    expiry_time - now
                } else {
                    Duration::ZERO
                };
                let final_wait = wait_duration + Duration::from_millis(100); // Add a small buffer
                warn!(
                    "Rate limiter enforcing wait of {:?} before next request",
                    final_wait
                );
                return final_wait;
            }
            let fallback_wait = Duration::from_secs(1); // Fallback wait time
            warn!(
                "Rate limiter enforcing fallback wait of {:?}",
                fallback_wait
            );
            return fallback_wait;
        }

        // Update state with the new request
        self.request_timestamps.push(now);
        self.token_counts.push(token_count);

        Duration::ZERO // No need to wait
    }
}
