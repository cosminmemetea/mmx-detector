#!/bin/bash

# Exit on error
set -e

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Error: Please activate the virtual environment first."
    echo "Run: source venv/bin/activate  # On Windows: venv\\Scripts\\activate"
    exit 1
fi

# Check if required tools are installed
for tool in isort black flake8; do
    if ! command -v $tool &> /dev/null; then
        echo "Error: $tool is not installed. Install it with 'pip install $tool'."
        exit 1
    fi
done

echo "Sorting imports with isort..."
isort . --check-only --diff || { echo "isort found issues. Running auto-fix..."; isort .; }

echo "Formatting code with black..."
black . --check || { echo "black found issues. Running auto-fix..."; black .; }

echo "Running linting with flake8..."
flake8 . --max-line-length=88 --extend-ignore=E203

echo "Linting and formatting complete!"