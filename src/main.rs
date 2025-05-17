use anyhow::{Context, Result};
use clap::Parser;
use dotenv::dotenv;
use log::{error, info};
use std::path::Path;

use gemini_rag::chunking;
use gemini_rag::database::{QdrantClient, QdrantConfig};
use gemini_rag::embeddings::{GeminiClient, GeminiConfig};
use gemini_rag::rag::RagEngine;

/// A RAG (Retrieval-Augmented Generation) application using Gemini embeddings and Qdrant
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Path to the text file to process
    #[arg(index = 1)]
    file_path: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize environment
    dotenv().ok();
    env_logger::init();

    // Parse and validate command line arguments
    let args = Args::parse();
    let file_path = args.file_path; // Path to the document to process

    info!("Processing file: {}", file_path);

    // Validate input file exists
    let path = Path::new(&file_path);
    if !path.exists() {
        error!("File not found: {}", file_path);
        return Err(anyhow::anyhow!("File not found"));
    }

    // Use filename as collection name in Qdrant
    let file_name = path.file_name().context("Invalid file path")?;

    // Load configuration from environment
    let qdrant_config = QdrantConfig::from_env().context("Missing QDRANT_URL")?;
    let gemini_config = GeminiConfig::from_env().context("Missing GEMINI_API_KEY")?;

    let qdrant = QdrantClient::new(qdrant_config)
        .await
        .context("Failed to initialize Qdrant client")?;
    let gemini = GeminiClient::new(gemini_config);

    // Initialize RAG engine
    let rag_engine = RagEngine::new(qdrant, gemini);

    // Convert filename to string for processing
    let file_name = file_name.to_str().context("Invalid file name")?;

    // Only process file if collection doesn't exist
    if rag_engine.collection_exists(file_name).await? {
        println!("Using existing collection: {}", file_name);
    } else {
        // Process and index the document
        let content = chunking::read_file(&file_path).context("Failed to read file")?;
        rag_engine
            .process_file(content, file_name)
            .await
            .context("Failed to process file")?;
    }

    // Enter interactive Q&A loop
    rag_engine
        .run_query_loop(file_name)
        .await
        .context("Error in query loop")?;

    Ok(())
}
