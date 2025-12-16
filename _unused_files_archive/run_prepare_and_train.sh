#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [ -d "$DIR/venv" ]; then
    echo "📦 Preparing Data..."
    "$DIR/venv/bin/python3" "$DIR/prepare_classification_data.py"
    
    echo "🧠 Launching Classification Training..."
    "$DIR/venv/bin/python3" "$DIR/train_classifier.py"
else
    echo "❌ Error: venv not found."
fi
