#!/bin/bash

# Exit on error
set -e

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Please activate the virtual environment first."
    echo "Run: source venv/bin/activate  # On Windows: venv\Scripts\activate"
    exit 1
fi

echo "Sorting imports with isort..."
isort .

echo "Formatting code with black..."
black .

echo "Running linting with flake8..."
flake8 .

echo "Linting and formatting complete!"