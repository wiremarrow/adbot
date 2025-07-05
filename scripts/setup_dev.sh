#!/bin/bash

# AdBot Development Environment Setup Script

set -e

echo "==================================="
echo "AdBot Development Environment Setup"
echo "==================================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.11"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)"; then
    echo "Error: Python 3.11+ is required. Current version: $python_version"
    exit 1
fi

echo "✓ Python version check passed: $python_version"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Install package in development mode
echo "Installing AdBot in development mode..."
pip install -e .

# Create necessary directories
echo "Creating project directories..."
mkdir -p logs data/{raw,processed,interim} models mlruns outputs checkpoints

# Copy environment file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "✓ .env file created. Please update it with your API credentials."
else
    echo "✓ .env file already exists"
fi

# Install pre-commit hooks
echo "Installing pre-commit hooks..."
pre-commit install

# Initialize database (if PostgreSQL is running)
if command -v psql &> /dev/null; then
    echo "PostgreSQL detected. Would you like to initialize the database? (y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        python scripts/init_db.py
    fi
else
    echo "PostgreSQL not detected. Skipping database initialization."
fi

echo ""
echo "================================"
echo "Setup completed successfully! 🎉"
echo "================================"
echo ""
echo "Next steps:"
echo "1. Update .env file with your API credentials"
echo "2. Activate the virtual environment: source venv/bin/activate"
echo "3. Start the development server: uvicorn src.api.main:app --reload"
echo "4. Access the API documentation: http://localhost:8000/docs"
echo ""
echo "For more information, see the README.md file."