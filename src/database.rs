use crate::chunking::TextChunk;
use crate::embeddings::Embedding;
use anyhow::{Context, Result};
use qdrant_client::qdrant::UpsertPointsBuilder;
use qdrant_client::qdrant::{CreateCollectionBuilder, Distance, PointStruct, Value, VectorParams};
use qdrant_client::Qdrant;
use serde_json::json;
use std::collections::HashMap;
use std::env;

const COLLECTION_VECTOR_SIZE: u64 = 768; // Default dimension for most embedding models

/// Configuration for Qdrant
pub struct QdrantConfig {
    pub url: String,
    pub api_key: Option<String>,
}

impl QdrantConfig {
    /// Create a new configuration from environment variables
    pub fn from_env() -> Result<Self> {
        let url = env::var("QDRANT_URL")?;
        let api_key = env::var("QDRANT_API_KEY").ok();

        Ok(QdrantConfig { url, api_key })
    }
}

/// Client for interacting with Qdrant
pub struct QdrantClient {
    client: Qdrant,
}

impl QdrantClient {
    /// Create a new Qdrant client
    pub async fn new(config: QdrantConfig) -> Result<Self> {
        let config_builder = Qdrant::from_url(&config.url);
        let config_builder = if let Some(api_key) = config.api_key {
            config_builder.api_key(api_key)
        } else {
            config_builder
        };

        let client = config_builder.build()?;

        Ok(QdrantClient { client })
    }

    /// Check if a collection exists
    pub async fn collection_exists(&self, file_name: &str) -> Result<bool> {
        let collection_name = get_collection_name(file_name);

        match self.client.collection_info(&collection_name).await {
            Ok(_) => Ok(true),
            Err(qdrant_client::QdrantError::ResponseError { status })
                if status.code() == tonic::Code::NotFound =>
            {
                Ok(false)
            }
            Err(e) => Err(anyhow::anyhow!(
                "Failed to check collection existence: {}",
                e
            )),
        }
    }

    /// Create a new collection for a file
    pub async fn create_collection(&self, file_name: &str) -> Result<()> {
        let collection_name = get_collection_name(file_name);

        let create_collection = CreateCollectionBuilder::new(collection_name.clone())
            .vectors_config(VectorParams {
                size: COLLECTION_VECTOR_SIZE,
                distance: Distance::Cosine.into(),
                ..Default::default()
            });

        self.client
            .create_collection(create_collection)
            .await
            .with_context(|| format!("Failed to create collection {}", collection_name))?;

        Ok(())
    }

    /// Delete a collection
    pub async fn delete_collection(&self, file_name: &str) -> Result<()> {
        let collection_name = get_collection_name(file_name);

        self.client
            .delete_collection(collection_name.clone())
            .await
            .with_context(|| format!("Failed to delete collection {}", collection_name))?;

        Ok(())
    }

    /// Store chunks in the collection
    pub async fn store_chunks(
        &self,
        chunks: Vec<TextChunk>,
        embeddings: Vec<Embedding>,
        file_name: &str,
    ) -> Result<()> {
        let collection_name = get_collection_name(file_name);

        // Convert chunks and embeddings to points
        let points: Vec<PointStruct> = chunks
            .into_iter()
            .zip(embeddings.into_iter())
            .enumerate()
            .map(|(idx, (chunk, embedding))| {
                let payload: HashMap<String, Value> = serde_json::from_value(json!({
                    "text": chunk.text,
                    "document_id": chunk.document_id,
                    "start_position": chunk.start_position,
                    "chunk_index": idx,
                }))
                .unwrap();

                PointStruct::new(idx as u64, embedding.values, payload)
            })
            .collect();

        // Instead of directly passing the collection name, use the builder
        let upsert_request = UpsertPointsBuilder::new(collection_name.clone(), points).build();

        // Upsert points in batch
        self.client
            .upsert_points(upsert_request)
            .await
            .with_context(|| {
                format!("Failed to upsert points in collection {}", collection_name)
            })?;

        Ok(())
    }

    /// Search for relevant chunks
    pub async fn search(
        &self,
        query_embedding: Embedding,
        file_name: &str,
        limit: u64,
    ) -> Result<Vec<TextChunk>> {
        use qdrant_client::qdrant::{with_payload_selector, SearchPoints, WithPayloadSelector};

        let collection_name = get_collection_name(file_name);

        // Create search request
        let search_request = SearchPoints {
            collection_name: collection_name.clone(),
            vector: query_embedding.values,
            limit,
            with_payload: Some(WithPayloadSelector {
                selector_options: Some(with_payload_selector::SelectorOptions::Enable(true)),
            }),
            ..Default::default()
        };

        // Execute search
        let search_response = self
            .client
            .search_points(search_request)
            .await
            .with_context(|| format!("Failed to search collection {}", collection_name))?;

        // Convert search results back to TextChunks
        let chunks = search_response
            .result
            .into_iter()
            .filter_map(|scored_point| {
                let payload = scored_point.payload;
                let text = payload.get("text")?.as_str()?;
                // Get document_id from payload or fallback to file_name
                let document_id = payload
                    .get("document_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or(&file_name.to_string())
                    .to_string();

                // Get start position or default to 0
                let start_position = payload
                    .get("start_position")
                    .and_then(|v| v.as_integer())
                    .map(|v| v as usize)
                    .unwrap_or(0);

                Some(TextChunk {
                    text: text.to_string(),
                    token_count: text.split_whitespace().count(), // Estimate token count
                    document_id,
                    start_position,
                })
            })
            .collect();

        Ok(chunks)
    }
}

/// Generate a collection name from a file name
fn get_collection_name(file_name: &str) -> String {
    // Replace non-alphanumeric characters with underscores and convert to lowercase
    let name = file_name
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '_' })
        .collect::<String>()
        .to_lowercase();

    format!("rag_{}", name)
}
