# Default target (run with just `just`)
default:
    just --list

# Format code using rustfmt
format:
    cargo fmt

# Lint code using clippy
lint:
    cargo clippy -- -D warnings

# Build the project in release mode
build:
    cargo build --release

# Run tests
test:
    cargo test

# Run the application
run:
    cargo run

# Clean build artifacts
clean:
    cargo clean