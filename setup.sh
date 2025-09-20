#!/bin/bash

# HifazatAI Development Environment Setup

echo "Setting up HifazatAI development environment..."

# Create Python virtual environment for backend
echo "Creating Python virtual environment..."
cd backend
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Download spaCy model
echo "Downloading spaCy English model..."
python -m spacy download en_core_web_sm

echo "Backend setup complete!"

# Setup frontend
echo "Setting up frontend..."
cd ../frontend

# Install Node.js dependencies (already done, but just in case)
npm install

echo "Frontend setup complete!"

echo ""
echo "Setup complete! To start development:"
echo "1. Backend: cd backend && source venv/bin/activate && uvicorn api.main:app --reload"
echo "2. Frontend: cd frontend && npm run dev"
echo "3. Or use Docker: docker-compose up"