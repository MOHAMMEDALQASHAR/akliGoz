#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [ -d "$DIR/venv" ]; then
    echo "Creating Currency Templates..."
    "$DIR/venv/bin/python3" "$DIR/capture_templates.py"
else
    echo "Error: venv not found. Please set up the environment first."
fi
