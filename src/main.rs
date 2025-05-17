use anyhow::{Context, Result};
use clap::Parser;
use dotenv::dotenv;
use log::{info, error};
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
    
    // Parse command line arguments
    let args = Args::parse();
    let file_path = args.file_path;
    
    info!("Processing file: {}", file_path);
    
    // Check if the file exists
    let path = Path::new(&file_path);
    if !path.exists() {
        error!("File not found: {}", file_path);
        return Err(anyhow::anyhow!("File not found"));
    }
    
    // Extract file name for collection naming
    let file_name = path
        .file_name()
        .context("Invalid file path")?;
    
    // Initialize clients
    let qdrant_config = QdrantConfig::from_env()
        .context("Failed to load Qdrant configuration")?;
    let gemini_config = GeminiConfig::from_env()
        .context("Failed to load Gemini configuration")?;
    
    let qdrant = QdrantClient::new(qdrant_config).await
        .context("Failed to initialize Qdrant client")?;
    let gemini = GeminiClient::new(gemini_config);
    
    // Initialize RAG engine
    let rag_engine = RagEngine::new(qdrant, gemini);

    // todo: check of the file already in qdrant and skip that step
    // Read file content
    let content = chunking::read_file(&file_path)
        .context("Failed to read file")?;
    
    // Process file: chunk it, generate embeddings, and store in Qdrant (if needed)
    let file_name = file_name.to_str().context("Invalid file name")?;
    rag_engine.process_file(content, file_name).await
        .context("Failed to process file")?;
    
    // Start the query loop
    rag_engine.run_query_loop(file_name).await
        .context("Error in query loop")?;
    
    Ok(())
}
