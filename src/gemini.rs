use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::env;

/// Configuration for Gemini API
#[derive(Clone)]
pub struct GeminiConfig {
    pub api_key: String,
    pub base_url: String,
    pub embedding_model: String,
    pub generate_model: String,
    pub contextualize_model: String,
}

impl GeminiConfig {
    /// Create a new configuration from environment variables
    pub fn from_env() -> Result<Self> {
        let api_key = env::var("GEMINI_API_KEY")?;
        let base_url = env::var("GEMINI_BASE_URL").expect("GEMINI_BASE_URL not set");
        
        // Default models if not specified
        let embedding_model = env::var("EMBEDDING_MODEL")
            .unwrap_or_else(|_| "models/text-embedding-004".to_string());
        let generate_model = env::var("GENERATE_MODEL")
            .unwrap_or_else(|_| "models/gemini-2.5-flash-preview-05-20".to_string());
        let contextualize_model = env::var("CONTEXTUALIZE_MODEL")
            .unwrap_or_else(|_| "models/gemini-2.0-flash-lite".to_string());

        Ok(GeminiConfig {
            api_key,
            base_url,
            embedding_model,
            generate_model,
            contextualize_model,
        })
    }
}

/// Client for interacting with Gemini API
#[derive(Clone)]
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

    /// Get the client configuration
    pub fn config(&self) -> &GeminiConfig {
        &self.config
    }

    /// Generate embeddings for a text
    pub async fn get_embedding(&self, text: &str) -> Result<Embedding> {
        #[derive(Serialize)]
        struct EmbeddingContent<'a> {
            parts: Vec<Part<'a>>,
        }

        #[derive(Serialize)]
        struct EmbeddingRequest<'a> {
            model: &'a str,
            content: EmbeddingContent<'a>,
        }

        let request = EmbeddingRequest {
            model: &self.config.embedding_model,
            content: EmbeddingContent {
                parts: vec![Part { text }],
            },
        };

        let url = format!("{}/{}:embedContent?key={}", 
            self.config.base_url, 
            self.config.embedding_model,
            self.config.api_key);

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

    /// Generate text using Gemini model
    pub async fn generate_text(
        &self,
        prompt: &str,
        model: &str,
        temperature: f32,
        top_p: f32,
        top_k: i32,
        max_output_tokens: i32,
    ) -> Result<String> {
        let request = GenerateRequest {
            model,
            contents: vec![Content::new_with_role(prompt, "user")],
            generation_config: GenerationConfig {
                temperature,
                top_p,
                top_k,
                max_output_tokens,
            },
        };

        let url = format!("{}/{}:generateContent?key={}", 
            self.config.base_url, 
            model, // Use the model parameter
            self.config.api_key);

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

    /// Generate a response based on context and question
    /// Uses Gemini 2.5 Flash Preview 05-20 by default for question answering
    pub async fn generate_answer(&self, context: &str, question: &str) -> Result<String> {
        let prompt = format!("Context: {}\n\nQuestion: {}", context, question);

        self.generate_text(
            &prompt,
            &self.config.generate_model,
            0.2,
            0.8,
            40,
            1024,
        )
        .await
    }

    /// Generate context using Gemini 2.0 Flash-Lite model specifically for summarization
    pub async fn generate_context(&self, prompt: &str) -> Result<String> {
        self.generate_text(
            prompt,
            &self.config.contextualize_model,
            0.2,
            0.8,
            40,
            512, // Shorter output for context generation
        )
        .await
    }
}

/// Representation of a vector embedding
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Embedding {
    pub values: Vec<f32>,
}

// Shared request/response structures for the Gemini API

// EmbeddingRequest struct is defined inline in get_embedding method

#[derive(Deserialize, Debug)]
struct EmbeddingResponse {
    embedding: EmbeddingData,
}

#[derive(Deserialize, Debug)]
struct EmbeddingData {
    values: Vec<f32>,
}

#[derive(Serialize)]
struct GenerateRequest<'a> {
    model: &'a str,
    contents: Vec<Content<'a>>,
    generation_config: GenerationConfig,
}

#[derive(Serialize)]
struct Content<'a> {
    parts: Vec<Part<'a>>,
    role: &'static str,
}

impl<'a> Content<'a> {
    fn new_with_role(text: &'a str, role: &'static str) -> Self {
        Content {
            parts: vec![Part { text }],
            role,
        }
    }
}

#[derive(Serialize)]
struct Part<'a> {
    text: &'a str,
}

#[derive(Serialize)]
struct GenerationConfig {
    temperature: f32,
    top_p: f32,
    top_k: i32,
    max_output_tokens: i32,
}

#[derive(Deserialize, Debug)]
struct GenerateResponse {
    candidates: Vec<Candidate>,
}

#[derive(Deserialize, Debug)]
struct Candidate {
    content: ResponseContent,
}

#[derive(Deserialize, Debug)]
struct ResponseContent {
    parts: Vec<ResponsePart>,
}

#[derive(Deserialize, Debug)]
struct ResponsePart {
    text: String,
}
