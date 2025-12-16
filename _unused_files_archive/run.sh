#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "=========================================="
echo "    👓 SMART GLASSES SYSTEM LAUNCHER 👓   "
echo "=========================================="

# Check for VENV
if [ -d "$DIR/venv" ]; then
    echo "✅ Virtual Environment Found!"
    PYTHON_EXEC="$DIR/venv/bin/python3"
else
    echo "⚠️  Virtual Environment NOT found. Using system Python."
    PYTHON_EXEC="python3"
fi

echo "🚀 Executing with: $PYTHON_EXEC"
"$PYTHON_EXEC" "$DIR/main_glasses.py"

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "------------------------------------------"
    echo "❌ App crashed with error code: $EXIT_CODE"
    echo "   If you see 'ModuleNotFoundError', please run:"
    echo "   ./venv/bin/pip install -r requirements.txt"
    echo "------------------------------------------"
fi
