#!/bin/bash

# Start script for SageMaker inference server

if [ "${1:-}" = "serve" ]; then
    echo "Starting SageMaker inference server..."
    # Change to app directory and run the adapter; it will spawn llama-server internally
    cd /opt/app
    exec python3 -m app.main
else
    # Default to bash if no arguments provided
    exec bash
fi
