#!/bin/bash
cd /app/blitz
# Activate the virtual environment (optional if using direct python path)
source .venv/bin/activate
# Run the application using exec to replace the shell process
exec python -m blitz
