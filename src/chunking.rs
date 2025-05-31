use anyhow::Result;
use std::fs;
use std::path::Path;

/// Represents a text chunk with metadata
#[derive(Debug, Clone)]
pub struct TextChunk {
    /// The actual text content of this chunk
    pub text: String,
    /// Estimated token count for this chunk
    pub token_count: usize,
    /// Unique identifier for the document this chunk belongs to
    pub document_id: String,
    /// Starting position of this chunk in the original document
    pub start_position: usize,
}

/// Read content from a text file
pub fn read_file<P: AsRef<Path>>(file_path: P) -> Result<String> {
    let content = fs::read_to_string(file_path)?;
    Ok(content)
}

/// Split text into chunks of approximately 500 tokens
pub fn split_into_chunks(text: &str, file_name: &str) -> Vec<TextChunk> {
    const TARGET_TOKENS: usize = 500;
    const OVERLAP_TOKENS: usize = 50; // Overlap between chunks for context

    // First, split by paragraphs
    let paragraphs: Vec<&str> = text
        .split("\n\n")
        .filter(|p| !p.trim().is_empty())
        .collect();

    let mut chunks = Vec::new();
    let mut current_chunk = String::new();
    let mut current_token_count = 0;

    // Process each paragraph
    for paragraph in paragraphs {
        let paragraph = paragraph.trim();

        // Estimate token count for the paragraph
        let paragraph_token_count = estimate_token_count(paragraph);

        // If a single paragraph is too large, split it into sentences
        if paragraph_token_count > TARGET_TOKENS {
            // Split into sentences (naive split on punctuation)
            let sentences: Vec<&str> = paragraph
                .split(|c| ".!?\n".contains(c))
                .filter(|s| !s.trim().is_empty())
                .collect();

            let mut sentence_buffer = String::new();
            let mut buffer_token_count = 0;

            for sentence in sentences {
                let sentence = sentence.trim();
                if sentence.is_empty() {
                    continue;
                }

                let sentence_token_count = estimate_token_count(sentence);

                // If adding this sentence would exceed the token limit
                if buffer_token_count + sentence_token_count > TARGET_TOKENS
                    && !sentence_buffer.is_empty()
                {
                    // Add the current buffer as a chunk
                    let start_position = text.find(&sentence_buffer).unwrap_or(0);
                    chunks.push(TextChunk {
                        text: sentence_buffer.clone(),
                        token_count: buffer_token_count,
                        document_id: file_name.to_string(),
                        start_position,
                    });

                    // Start a new buffer with overlap from the previous chunk
                    let overlap_start = sentence_buffer
                        .char_indices()
                        .nth(
                            sentence_buffer
                                .chars()
                                .count()
                                .saturating_sub(OVERLAP_TOKENS * 4),
                        ) // Approximate char count for overlap tokens
                        .map(|(i, _)| i)
                        .unwrap_or(0);

                    sentence_buffer = sentence_buffer[overlap_start..].trim().to_string();
                    buffer_token_count = estimate_token_count(&sentence_buffer);
                }

                // Add the current sentence to the buffer
                if !sentence_buffer.is_empty() {
                    sentence_buffer.push(' ');
                }
                sentence_buffer.push_str(sentence);
                sentence_buffer.push('.'); // Add back the period
                buffer_token_count += sentence_token_count + 1; // +1 for the period
            }

            // Add any remaining content in the buffer
            if !sentence_buffer.is_empty() {
                let start_position = text.find(&sentence_buffer).unwrap_or(0);
                chunks.push(TextChunk {
                    text: sentence_buffer.clone(),
                    token_count: buffer_token_count,
                    document_id: file_name.to_string(),
                    start_position,
                });
            }
        } else {
            // Check if adding this paragraph would exceed the token limit
            if current_token_count + paragraph_token_count > TARGET_TOKENS
                && !current_chunk.is_empty()
            {
                // Current chunk would exceed token limit, so finalize it
                let start_position = text.find(&current_chunk).unwrap_or(0);
                chunks.push(TextChunk {
                    text: current_chunk.clone(),
                    token_count: current_token_count,
                    document_id: file_name.to_string(),
                    start_position,
                });

                // Start a new chunk with overlap from the previous chunk
                let overlap_start = current_chunk
                    .char_indices()
                    .nth(
                        current_chunk
                            .chars()
                            .count()
                            .saturating_sub(OVERLAP_TOKENS * 4),
                    ) // Approximate char count for overlap tokens
                    .map(|(i, _)| i)
                    .unwrap_or(0);

                current_chunk = current_chunk[overlap_start..].trim().to_string();
                current_token_count = estimate_token_count(&current_chunk);

                if !current_chunk.is_empty() {
                    current_chunk.push_str("\n\n");
                }
            }

            // Add the paragraph to the current chunk
            if !current_chunk.is_empty() && !current_chunk.ends_with("\n\n") {
                current_chunk.push_str("\n\n");
            }
            current_chunk.push_str(paragraph);
            current_token_count += paragraph_token_count;
        }
    }

    // Add the last chunk if it's not empty
    if !current_chunk.trim().is_empty() {
        let start_position = text.find(&current_chunk).unwrap_or(0);
        chunks.push(TextChunk {
            text: current_chunk,
            token_count: current_token_count,
            document_id: file_name.to_string(),
            start_position,
        });
    }

    // Ensure no chunk is too large
    let mut final_chunks = Vec::new();
    for chunk in chunks {
        if chunk.token_count > TARGET_TOKENS * 3 {
            // If a chunk is still too large, split it by sentences
            let TextChunk {
                text,
                token_count: _,
                document_id,
                start_position: _,
            } = chunk;
            // Recursively split into chunks
            let mut sub_chunks = split_into_chunks(&text, &document_id);
            // Ensure document_id is preserved in sub-chunks
            for sub_chunk in &mut sub_chunks {
                sub_chunk.document_id = document_id.clone();
            }
            final_chunks.append(&mut sub_chunks);
        } else {
            final_chunks.push(chunk);
        }
    }

    final_chunks
}

/// Calculate approximate token count for a text
/// This is a very simple estimation - words plus punctuation
pub fn estimate_token_count(text: &str) -> usize {
    // Simple approximation: count words and punctuation marks
    let words = text.split_whitespace().count();
    let punctuation = text.chars().filter(|c| c.is_ascii_punctuation()).count();
    words + punctuation
}
