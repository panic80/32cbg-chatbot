#!/bin/bash
cd /Users/mattermost/01_Active_Work/Current_Projects/testtest/ujjval/32cbg-chatbot/rag-service
source venv/bin/activate
pkill -f uvicorn 2>/dev/null || true
sleep 2
uvicorn app.main:app --host 0.0.0.0 --port 8000 &
UVICORN_PID=$!
echo "Started uvicorn with PID: $UVICORN_PID"
sleep 12
echo "Checking health..."
curl -s http://localhost:8000/api/v1/health
echo ""
echo "Killing uvicorn..."
kill $UVICORN_PID 2>/dev/null || true
