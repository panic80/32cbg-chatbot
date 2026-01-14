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
            │   (cache)   │  │  (FastAPI)  │  │     API     │
            └─────────────┘  └──────┬──────┘  └─────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
            ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
            │  ChromaDB   │  │   OpenAI    │  │  Anthropic  │
            │  (vectors)  │  │   Gemini    │  │ OpenRouter  │
            └─────────────┘  └─────────────┘  └─────────────┘
```

### Components

| Component | Technology | Directory | Purpose |
|-----------|------------|-----------|---------|
| Frontend | React 18 + Vite + TailwindCSS | `src/` | Chat UI, admin panels, trip planner |
| Backend Gateway | Node.js + Express | `server/` | API routing, auth, rate limiting, caching |
| RAG Service | Python FastAPI | `rag-service/` | Document retrieval, embeddings, LLM orchestration |
| Vector Store | ChromaDB | `rag-service/chroma_db/` | Semantic search over policy documents |
| Cache | Redis | - | Response caching, session state |

## Prerequisites

- Node.js 18+ and npm
- Python 3.10+ (for RAG service)
- Redis (optional, for caching)
- Docker & Docker Compose (optional, for containerized deployment)

## Quick Start

### Local Development

1. **Install dependencies:**
   ```bash
   npm install
   ```

2. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Start development servers:**
   ```bash
   npm run dev:full
   ```
   Or use the helper script:
   ```bash
   ./start-dev.sh
   ```

4. **Access the application:**
   - Frontend: http://localhost:3001
   - Backend API: http://localhost:3000
   - Health check: http://localhost:3000/health

### Docker Deployment

```bash
# Start all services
./docker-start.sh

# Stop services
./docker-stop.sh
```

Docker exposes:
- Application: http://localhost:3000
- RAG Service: http://localhost:8000

## Configuration

Copy `.env.example` to `.env` and configure the following:

### Required

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key for GPT models |
| `CONFIG_PANEL_USER` | Admin panel username |
| `CONFIG_PANEL_PASSWORD` | Admin panel password |
| `ADMIN_API_TOKEN` | Token for admin API endpoints |

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
| `REDIS_URL` | `redis://localhost:6379` | Redis connection URL |
| `ENABLE_CACHE` | `true` | Enable Redis caching |
| `CACHE_TTL` | `3600000` | Cache TTL in milliseconds |
| `RAG_SERVICE_URL` | `http://localhost:8000` | RAG service URL |
| `GOOGLE_MAPS_API_KEY` | - | Google Maps API for trip planning |
| `TRIP_PLANNER_MODEL` | `gpt-4.1-mini` | Model for trip planning |
| `ENABLE_RATE_LIMIT` | `true` | Enable API rate limiting |
| `RATE_LIMIT_MAX` | `60` | Max requests per window |
| `RATE_LIMIT_WINDOW` | `60000` | Rate limit window (ms) |

For production, store sensitive values in `/etc/cbthis/env`.

## Available Scripts

| Script | Description |
|--------|-------------|
| `npm run dev` | Start Vite dev server (frontend only) |
| `npm run dev:server` | Start Express server (backend only) |
| `npm run dev:full` | Start both frontend and backend |
| `npm run build` | Build server and frontend for production |
| `npm start` | Run production server |
| `npm test` | Run tests with Vitest |
| `npm run test:watch` | Run tests in watch mode |
| `npm run test:coverage` | Run tests with coverage report |
| `npm run lint` | Run ESLint and Prettier checks |

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

### OPI Directory (`/opi`)
- Office of Primary Interest contact list
- Finance personnel points of contact

### Performance Dashboard (`/admin/performance`)
- System metrics and monitoring
- API response times
- Cache hit rates

### Resources (`/resources`)
- Links to official CFTI documentation
- Reference materials

## API Endpoints

### Public
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/chat` | POST | Send chat message |
| `/api/chat/stream` | POST | Streaming chat |

### Admin (requires authentication)
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/admin/*` | * | Admin operations |
| `/api/config/*` | * | Configuration management |
| `/api/sources/*` | * | Source document management |
| `/api/ingestion/*` | * | Document ingestion |
| `/api/analytics/*` | GET | Usage analytics |
| `/api/performance/*` | GET | Performance metrics |

## RAG Service

The RAG (Retrieval-Augmented Generation) service handles document retrieval and LLM interactions.

### Setup
```bash
cd rag-service
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Document Ingestion
```bash
# From project root
python ingest_sources_cli.py --help
```

See `rag-service/QUICK_START_STATEFUL.md` for detailed setup instructions.

## Testing

```bash
# Run all tests
npm test

# Watch mode
npm run test:watch

# Coverage report
npm run test:coverage
```

## Deployment

### PM2 (Production)
```bash
# Setup
npm run deploy:setup

# Deploy
npm run deploy:production

# Rollback
npm run rollback:production
```

### Docker
```bash
docker-compose up -d
```

## Project Structure

```
.
├── src/                    # React frontend
│   ├── components/         # Reusable UI components
│   ├── pages/              # Page components
│   ├── hooks/              # Custom React hooks
│   ├── context/            # React context providers
│   ├── services/           # API client services
│   └── utils/              # Utility functions
├── server/                 # Express backend
│   ├── routes/             # API route handlers
│   ├── middleware/         # Express middleware
│   ├── services/           # Business logic
│   └── config/             # Server configuration
├── rag-service/            # Python RAG service
│   ├── app/                # FastAPI application
│   ├── config/             # RAG configuration
│   └── data/               # Document sources
├── public/                 # Static assets
├── docs/                   # Documentation
└── scripts/                # Deployment scripts
```

## License

Proprietary - Canadian Forces internal use only.
