#!/bin/bash
echo "Starting CF Travel Bot in Docker..."
docker-compose up --build -d
echo "Services started."
echo "Frontend/Backend: http://localhost:3000"
echo "RAG Service:      http://localhost:8000"
echo ""
echo "To view logs: docker-compose logs -f"
