#!/bin/bash

# Canadian Forces Travel Instructions Chatbot - Development Server Startup
echo "🚀 Starting Canadian Forces Travel Instructions Chatbot..."

# Kill any existing processes on our ports
echo "🧹 Cleaning up existing processes..."
lsof -ti:3000 | xargs kill -9 2>/dev/null || true
lsof -ti:3001 | xargs kill -9 2>/dev/null || true

# Wait a moment for cleanup
sleep 2

# Start both servers concurrently
echo "🔧 Starting backend server on port 3000..."
echo "🔧 Starting Vite dev server on port 3001..."
echo ""
echo "📱 React App: http://localhost:3001"
echo "📱 Chat Interface: http://localhost:3001/chat"
echo "🔌 Backend API: http://localhost:3000"
echo "❤️  Health Check: http://localhost:3000/health"
echo ""
echo "Press Ctrl+C to stop all servers"
echo "----------------------------------------"

npm run dev:full