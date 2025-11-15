#!/bin/bash
echo "Setting up .env file..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "✅ Created .env file from .env.example"
    echo "📝 Please edit .env and add your OPENAI_API_KEY"
    echo "   Get your key from: https://platform.openai.com/api-keys"
else
    echo "⚠️  .env file already exists"
fi
