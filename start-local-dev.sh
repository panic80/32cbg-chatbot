#!/bin/bash

# ============================================================================
# LOCAL DEV MODE - Start all services without Docker
# ============================================================================
# Architecture:
#   Frontend (Vite)     -> localhost:3001
#   Node.js Backend     -> localhost:3000
#   Python RAG Service  -> localhost:8000 (venv)
#   Redis               -> DISABLED (in-memory fallbacks)
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAG_DIR="$SCRIPT_DIR/rag-service"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  LOCAL DEV MODE - No Docker, No Redis${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# ============================================================================
# Pre-flight checks
# ============================================================================

# Check if .env exists
if [ ! -f "$SCRIPT_DIR/.env" ]; then
    echo -e "${YELLOW}No .env file found. Creating from .env.example...${NC}"
    cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env"
    echo -e "${YELLOW}Please edit .env and add your API keys, then run this script again.${NC}"
    exit 1
fi

# Check if RAG venv exists
if [ ! -d "$RAG_DIR/venv" ]; then
    echo -e "${RED}RAG service venv not found!${NC}"
    echo -e "${YELLOW}Run this first: cd rag-service && ./setup.sh${NC}"
    exit 1
fi

# Check Node.js
if ! command -v node &> /dev/null; then
    echo -e "${RED}Node.js is not installed!${NC}"
    exit 1
fi

# Check if node_modules exists
if [ ! -d "$SCRIPT_DIR/node_modules" ]; then
    echo -e "${YELLOW}Node modules not found. Installing...${NC}"
    npm install
fi

# ============================================================================
# Kill existing processes
# ============================================================================

echo -e "${YELLOW}Cleaning up existing processes...${NC}"
lsof -ti:3000 | xargs kill -9 2>/dev/null || true
lsof -ti:3001 | xargs kill -9 2>/dev/null || true
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
sleep 1

# ============================================================================
# Start RAG Service (Python)
# ============================================================================

echo -e "${GREEN}Starting Python RAG Service on port 8000...${NC}"
(
    cd "$RAG_DIR"
    source venv/bin/activate

    # Export env vars for RAG service (disable Redis)
    export REDIS_URL=""

    # Start uvicorn with auto-reload
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 2>&1 | sed 's/^/[RAG] /' &
)

# Wait for RAG service to be healthy
echo -e "${YELLOW}Waiting for RAG service to start...${NC}"
for i in {1..30}; do
    if curl -s http://localhost:8000/api/v1/health > /dev/null 2>&1; then
        echo -e "${GREEN}RAG service is healthy!${NC}"
        break
    fi
    if [ $i -eq 30 ]; then
        echo -e "${RED}RAG service failed to start. Check logs above.${NC}"
        exit 1
    fi
    sleep 1
done

# ============================================================================
# Start Node.js Backend + Vite Frontend
# ============================================================================

echo -e "${GREEN}Starting Node.js backend (port 3000) and Vite frontend (port 3001)...${NC}"

# Export env vars to disable Redis for Node.js
export REDIS_URL=""
export ENABLE_CACHE=false
export RAG_SERVICE_URL=http://localhost:8000

# Start Node.js dev servers (backend + frontend)
npm run dev:full &

# Wait a moment for servers to start
sleep 3

# ============================================================================
# Summary
# ============================================================================

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  All services started!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo -e "  ${BLUE}Frontend:${NC}     http://localhost:3001"
echo -e "  ${BLUE}Chat:${NC}         http://localhost:3001/chat"
echo -e "  ${BLUE}Backend API:${NC}  http://localhost:3000"
echo -e "  ${BLUE}RAG API:${NC}      http://localhost:8000"
echo -e "  ${BLUE}RAG Docs:${NC}     http://localhost:8000/api/v1/docs"
echo ""
echo -e "  ${YELLOW}Redis:${NC}        DISABLED (using in-memory fallbacks)"
echo ""
echo -e "Press ${RED}Ctrl+C${NC} to stop all servers"
echo -e "${GREEN}============================================${NC}"

# Keep script running and forward Ctrl+C to children
trap "echo ''; echo 'Stopping all services...'; kill 0" EXIT
wait
