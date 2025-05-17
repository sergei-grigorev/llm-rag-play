use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::env;

/// Configuration for Gemini API
pub struct GeminiConfig {
    pub api_key: String,
    pub embeddings_url: String,
    pub generate_url: String,
}

/// Representation of a vector embedding
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Embedding {
    pub values: Vec<f32>,
}

impl GeminiConfig {
    /// Create a new configuration from environment variables
    pub fn from_env() -> Result<Self> {
        let api_key = env::var("GEMINI_API_KEY")?;
        // Default URLs if not specified
        let embeddings_url =
            env::var("GEMINI_EMBEDDINGS_URL").expect("GEMINI_EMBEDDINGS_URL not set");
        let generate_url = env::var("GEMINI_GENERATE_URL").expect("GEMINI_GENERATE_URL not set");

        Ok(GeminiConfig {
            api_key,
            embeddings_url,
            generate_url,
        })
    }
}

/// Client for interacting with Gemini API
pub struct GeminiClient {
    config: GeminiConfig,
    client: reqwest::Client,
}

impl GeminiClient {
    /// Create a new Gemini client
    pub fn new(config: GeminiConfig) -> Self {
        let client = reqwest::Client::new();
        GeminiClient { config, client }
    }

    /// Generate embeddings for a text
    pub async fn get_embedding(&self, text: &str) -> Result<Embedding> {
        #[derive(serde::Serialize)]
        struct EmbeddingRequest<'a> {
            model: &'static str,
            content: Content<'a>,
        }

        #[derive(serde::Serialize)]
        struct Content<'a> {
            parts: Vec<Part<'a>>,
        }

        #[derive(serde::Serialize)]
        struct Part<'a> {
            text: &'a str,
        }

        #[derive(serde::Deserialize, Debug)]
        struct EmbeddingResponse {
            embedding: EmbeddingData,
        }

        #[derive(serde::Deserialize, Debug)]
        struct EmbeddingData {
            values: Vec<f32>,
        }

        let request = EmbeddingRequest {
            model: "models/embedding-004",
            content: Content {
                parts: vec![Part { text }],
            },
        };

        let url = format!("{}?key={}", self.config.embeddings_url, self.config.api_key);

        let response = self.client.post(&url).json(&request).send().await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(anyhow::anyhow!(
                "API request failed: {} {}",
                status,
                error_text
            ));
        }

        let response_data: EmbeddingResponse = response.json().await?;

        Ok(Embedding {
            values: response_data.embedding.values,
        })
    }

    /// Generate a response based on context and question
    pub async fn generate_answer(&self, context: &str, question: &str) -> Result<String> {
        #[derive(serde::Serialize)]
        struct GenerateRequest<'a> {
            contents: Vec<Content<'a>>,
            generation_config: GenerationConfig,
        }

        #[derive(serde::Serialize)]
        struct Content<'a> {
            parts: Vec<Part<'a>>,
            role: &'static str,
        }

        #[derive(serde::Serialize)]
        struct Part<'a> {
            text: &'a str,
        }

        #[derive(serde::Serialize)]
        struct GenerationConfig {
            temperature: f32,
            top_p: f32,
            top_k: i32,
            max_output_tokens: i32,
        }

        #[derive(serde::Deserialize, Debug)]
        struct GenerateResponse {
            candidates: Vec<Candidate>,
        }

        #[derive(serde::Deserialize, Debug)]
        struct Candidate {
            content: ResponseContent,
        }

        #[derive(serde::Deserialize, Debug)]
        struct ResponseContent {
            parts: Vec<ResponsePart>,
        }

        #[derive(serde::Deserialize, Debug)]
        struct ResponsePart {
            text: String,
        }

        let prompt = format!("Context: {}\n\nQuestion: {}", context, question);

        let request = GenerateRequest {
            contents: vec![Content {
                parts: vec![Part { text: &prompt }],
                role: "user",
            }],
            generation_config: GenerationConfig {
                temperature: 0.2,
                top_p: 0.8,
                top_k: 40,
                max_output_tokens: 1024,
            },
        };

        let url = format!("{}?key={}", self.config.generate_url, self.config.api_key);

        let response = self.client.post(&url).json(&request).send().await?;

        if !response.status().is_success() {
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(anyhow::anyhow!("API request failed: {}", error_text));
        }

        let response_data: GenerateResponse = response.json().await?;

        // Extract the generated text from the response
        response_data
            .candidates
            .into_iter()
            .next()
            .and_then(|c| c.content.parts.into_iter().next())
            .map(|p| p.text)
            .ok_or_else(|| anyhow::anyhow!("No response generated"))
    }
}
