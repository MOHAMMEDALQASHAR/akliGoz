#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [ -d "$DIR/venv" ]; then
    echo "🎥 Launching Data Collector..."
    "$DIR/venv/bin/python3" "$DIR/capture_dataset.py"
else
    echo "❌ Error: venv not found."
fi
