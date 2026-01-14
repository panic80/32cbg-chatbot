#!/bin/bash

# Start the hardened gateway only (legacy proxy removed)
echo "Starting main server on port 3000..."
node dist-server/main.js &
MAIN_PID=$!

# Handle graceful shutdown
trap 'kill $MAIN_PID; exit' SIGINT SIGTERM

# Keep script running
wait
