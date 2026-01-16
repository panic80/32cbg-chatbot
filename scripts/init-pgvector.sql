-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Grant privileges to the rag_user
-- Note: The database and user are created by Docker's POSTGRES_USER/POSTGRES_DB env vars
-- This script just ensures the vector extension is enabled
