# CF Travel Bot

An AI-powered chatbot for Canadian Forces Travel Instructions (CFTI) policy guidance. The system provides conversational assistance for travel policy questions, trip planning support, and administrative tools for managing the knowledge base.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              Frontend                                    │
│                   Vite + React + TailwindCSS                            │
│                        (src/ - port 3001)                               │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          Backend Gateway                                 │
│                      Node.js + Express                                  │
│                     (server/ - port 3000)                               │
│   ┌─────────────┬───────────────┬──────────────┬─────────────────┐     │
│   │    Auth     │  Rate Limit   │   Caching    │   API Proxy     │     │
│   └─────────────┴───────────────┴──────────────┴─────────────────┘     │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
            ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
            │    Redis    │  │ RAG Service │  │ Google Maps │
            │  (optional) │  │  (FastAPI)  │  │     API     │
            └─────────────┘  └──────┬──────┘  └─────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
            ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
            │ PostgreSQL  │  │   OpenAI    │  │  Anthropic  │
            │ + pgvector  │  │   Gemini    │  │ OpenRouter  │
            └─────────────┘  └─────────────┘  └─────────────┘
```

### Components

| Component | Technology | Directory | Purpose |
|-----------|------------|-----------|---------|
| Frontend | React 18 + Vite + TailwindCSS | `src/` | Chat UI, admin panels, trip planner |
| Backend Gateway | Node.js + Express | `server/` | API routing, auth, rate limiting, caching |
| RAG Service | Python FastAPI | `rag-service/` | Document retrieval, embeddings, LLM orchestration |
| Vector Store | PostgreSQL + pgvector | - | Semantic search over policy documents |
| Cache | Redis (optional) | - | Response caching (disabled by default) |

## Prerequisites

- **Node.js 18+** and npm
- **Python 3.10+** (3.11 recommended)
- **PostgreSQL 14+** with pgvector extension
- Redis (optional, disabled by default)

---

## Local Development Setup

Follow these steps in order to get the project running locally.

### Step 1: Clone the Repository

```bash
git clone https://github.com/panic80/32cbg-chatbot.git
cd 32cbg-chatbot
```

### Step 2: Install PostgreSQL with pgvector

**macOS (Homebrew):**
```bash
brew install postgresql@14
brew install pgvector

# Start PostgreSQL
brew services start postgresql@14

# Create the database and enable pgvector
psql postgres -c "CREATE DATABASE rag_vectorstore;"
psql rag_vectorstore -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

**Ubuntu/Debian:**
```bash
sudo apt install postgresql postgresql-contrib
sudo apt install postgresql-14-pgvector

sudo systemctl start postgresql

sudo -u postgres psql -c "CREATE DATABASE rag_vectorstore;"
sudo -u postgres psql -d rag_vectorstore -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

**Verify pgvector is installed:**
```bash
psql rag_vectorstore -c "SELECT * FROM pg_extension WHERE extname = 'vector';"
```

### Step 3: Set Up the RAG Service (Python)

```bash
cd rag-service

# Run the setup script (creates venv, installs dependencies)
./setup.sh

# Or manually:
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Install Node.js Dependencies

```bash
cd ..  # Back to project root
npm install
```

### Step 5: Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```bash
# Required - at least one LLM provider
OPENAI_API_KEY=sk-your-openai-key-here

# Optional LLM providers
ANTHROPIC_API_KEY=your-anthropic-key
GEMINI_API_KEY=your-gemini-key

# Admin panel credentials
CONFIG_PANEL_USER=admin
CONFIG_PANEL_PASSWORD=your-secure-password
ADMIN_API_TOKEN=your-admin-token

# PostgreSQL (defaults work for local setup)
DATABASE_URL=postgresql://localhost:5432/rag_vectorstore

# Redis is DISABLED by default (uses in-memory fallbacks)
# REDIS_URL=redis://localhost:6379
# ENABLE_CACHE=false
```

### Step 6: Start All Services

**Option A: Use the helper script (recommended)**
```bash
./start-local-dev.sh
```

This script:
- Starts the Python RAG service (port 8000)
- Starts the Node.js backend (port 3000)
- Starts the Vite frontend (port 3001)
- Disables Redis (uses in-memory fallbacks)

**Option B: Start services manually**

Terminal 1 - RAG Service:
```bash
cd rag-service
source venv/bin/activate
uvicorn app.main:app --reload --port 8000
```

Terminal 2 - Backend + Frontend:
```bash
npm run dev:full
```

### Step 7: Verify Everything is Running

| Service | URL | What to Check |
|---------|-----|---------------|
| Frontend | http://localhost:3001 | Chat interface loads |
| Backend | http://localhost:3000/health | Returns `{"status":"healthy"}` |
| RAG Service | http://localhost:8000/api/v1/health | Returns health status |
| RAG Docs | http://localhost:8000/api/v1/docs | Swagger UI loads |

---

## Quick Reference

### Start Development
```bash
./start-local-dev.sh
```

### Stop All Services
Press `Ctrl+C` in the terminal running `start-local-dev.sh`

Or manually:
```bash
lsof -ti:3000,3001,8000 | xargs kill -9
```

### Rebuild After Code Changes

Frontend/Backend changes: Auto-reloads (hot module replacement)

RAG Service changes: Auto-reloads (uvicorn --reload)

After dependency changes:
```bash
# Node.js
npm install

# Python
cd rag-service && source venv/bin/activate && pip install -r requirements.txt
```

---

## Configuration

### Required Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key (required for embeddings) |
| `CONFIG_PANEL_USER` | Admin panel username |
| `CONFIG_PANEL_PASSWORD` | Admin panel password |

### LLM Providers (at least one required)

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI (GPT-4, GPT-3.5) |
| `ANTHROPIC_API_KEY` | Anthropic (Claude) |
| `GEMINI_API_KEY` | Google (Gemini) |
| `OPENROUTER_API_KEY` | OpenRouter (multi-provider) |

### Optional

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql://localhost:5432/rag_vectorstore` | PostgreSQL connection |
| `REDIS_URL` | `` (empty = disabled) | Redis connection URL |
| `ENABLE_CACHE` | `false` | Enable Redis caching |
| `RAG_SERVICE_URL` | `http://localhost:8000` | RAG service URL |
| `GOOGLE_MAPS_API_KEY` | - | Google Maps API for trip planning |

---

## Available Scripts

| Script | Description |
|--------|-------------|
| `npm run dev` | Start Vite dev server (frontend only) |
| `npm run dev:server` | Start Express server (backend only) |
| `npm run dev:full` | Start both frontend and backend |
| `npm run build` | Build server and frontend for production |
| `npm start` | Run production server |
| `npm test` | Run tests with Vitest |
| `npm run lint` | Run ESLint and Prettier checks |

---

## Features

### Chat Interface (`/chat`)
- Conversational AI for CFTI policy questions
- Multi-model support (GPT-4, Claude, Gemini)
- Source citations from policy documents
- Streaming responses

### Admin Tools (`/admin-tools`)
- Knowledge base management
- Document ingestion controls
- System configuration

### Configuration Panel (`/config`)
- Model selection and parameters
- Cache management
- System settings

---

## Troubleshooting

### "Port already in use" error
```bash
lsof -ti:3000,3001,8000 | xargs kill -9
```

### RAG service won't start
1. Check PostgreSQL is running: `pg_isready`
2. Check pgvector extension: `psql rag_vectorstore -c "\dx"`
3. Check venv is activated: `which python` should show `rag-service/venv/bin/python`

### "No module named 'app'" error
```bash
cd rag-service
source venv/bin/activate
pip install -e .
```

### Database connection refused
```bash
# macOS
brew services start postgresql@14

# Linux
sudo systemctl start postgresql
```

---

## Project Structure

```
.
├── src/                    # React frontend
│   ├── components/         # Reusable UI components
│   ├── pages/              # Page components
│   ├── hooks/              # Custom React hooks
│   ├── context/            # React context providers
│   └── services/           # API client services
├── server/                 # Express backend
│   ├── controllers/        # Request handlers
│   ├── routes/             # API route definitions
│   ├── middleware/         # Express middleware
│   └── services/           # Business logic
├── rag-service/            # Python RAG service
│   ├── app/                # FastAPI application
│   │   ├── api/            # API endpoints
│   │   ├── core/           # Core config, logging
│   │   ├── pipelines/      # Ingestion & retrieval
│   │   └── services/       # Business logic
│   ├── venv/               # Python virtual environment
│   └── requirements.txt    # Python dependencies
├── scripts/                # Helper scripts
├── start-local-dev.sh      # Local dev startup script
└── .env                    # Environment variables (create from .env.example)
```

---

## License

Proprietary - Canadian Forces internal use only.
