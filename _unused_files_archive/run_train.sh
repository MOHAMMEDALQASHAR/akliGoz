#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [ -d "$DIR/venv" ]; then
    echo "🧠 Launching AI Training..."
    "$DIR/venv/bin/python3" "$DIR/train_currency_model.py"
else
    echo "❌ Error: venv not found."
fi
