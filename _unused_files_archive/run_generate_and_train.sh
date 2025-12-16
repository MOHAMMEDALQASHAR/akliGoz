#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [ -d "$DIR/venv" ]; then
    echo "🔮 Generating Synthetic Data..."
    "$DIR/venv/bin/python3" "$DIR/generate_synthetic_data.py"
    
    echo "🧠 Launching AI Training..."
    "$DIR/venv/bin/python3" "$DIR/train_currency_model.py"
else
    echo "❌ Error: venv not found."
fi
