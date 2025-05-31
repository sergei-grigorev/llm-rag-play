use anyhow::{Context, Result};
use log::{debug, info, warn};
use mime_guess::from_path;
use pdf_extract::extract_text;
use std::fs;
use std::path::Path;

/// Represents a document with its content and metadata
#[derive(Debug, Clone)]
pub struct Document {
    /// The actual text content of the document
    pub content: String,
    /// The document's file name (used as document ID)
    pub document_id: String,
    /// The document's MIME type
    pub mime_type: String,
}

impl Document {
    /// Create a new document from a file path
    pub fn from_file<P: AsRef<Path>>(file_path: P) -> Result<Self> {
        let path = file_path.as_ref();
        let file_name = path
            .file_name()
            .context("Invalid file name")?
            .to_str()
            .context("Invalid file name encoding")?
            .to_string();

        // Detect MIME type
        let mime = from_path(path).first_or_octet_stream();
        let mime_type = mime.to_string();
        debug!("Detected MIME type: {}", mime_type);

        // Read content based on file type
        let content = read_document_content(path, &mime_type)?;

        Ok(Document {
            content,
            document_id: file_name,
            mime_type,
        })
    }
}

/// Read content from a document based on its MIME type
pub fn read_document_content<P: AsRef<Path>>(file_path: P, mime_type: &str) -> Result<String> {
    let path = file_path.as_ref();

    match mime_type {
        // Handle PDF documents
        mime if mime.starts_with("application/pdf") => {
            info!("Processing PDF document: {}", path.display());
            let content = extract_text(path)
                .with_context(|| format!("Failed to extract text from PDF: {}", path.display()))?;

            // PDF extraction can sometimes include excessive whitespace
            let cleaned_content = normalize_whitespace(&content);

            if cleaned_content.is_empty() {
                warn!("Extracted PDF content is empty or contains only whitespace");
            }

            Ok(cleaned_content)
        }

        // Handle plain text documents
        mime if mime.starts_with("text/") => {
            info!("Processing text document: {}", path.display());
            let content = fs::read_to_string(path)
                .with_context(|| format!("Failed to read text file: {}", path.display()))?;
            Ok(content)
        }

        // Unsupported format
        _ => Err(anyhow::anyhow!(
            "Unsupported document format: {}. Only text and PDF files are supported.",
            mime_type
        )),
    }
}

/// Normalize whitespace in text (remove multiple consecutive spaces, newlines, etc.)
fn normalize_whitespace(text: &str) -> String {
    // Replace multiple spaces with a single space
    let result = text.replace('\r', "");

    // Replace multiple consecutive newlines with double newlines (paragraph separator)
    let mut prev_char = ' ';
    let mut newline_count = 0;
    let mut normalized = String::with_capacity(result.len());

    for c in result.chars() {
        if c == '\n' {
            newline_count += 1;
        } else {
            if newline_count > 0 {
                // Add at most two newlines (paragraph break)
                if newline_count >= 2 {
                    normalized.push_str("\n\n");
                } else {
                    normalized.push('\n');
                }
                newline_count = 0;
            }

            // Don't add consecutive spaces
            if !(c == ' ' && prev_char == ' ') {
                normalized.push(c);
            }

            prev_char = c;
        }
    }

    // Handle trailing newlines
    if newline_count > 0 {
        if newline_count >= 2 {
            normalized.push_str("\n\n");
        } else {
            normalized.push('\n');
        }
    }

    normalized.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_whitespace() {
        let text = "This  has   multiple    spaces.\n\n\nAnd multiple newlines.\r\nAnd Windows line endings.";
        let expected =
            "This has multiple spaces.\n\nAnd multiple newlines.\nAnd Windows line endings.";
        assert_eq!(normalize_whitespace(text), expected);
    }
}
