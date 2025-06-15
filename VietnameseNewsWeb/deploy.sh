#!/bin/bash

# Vietnamese News Classification - Docker Deployment Script
echo "🇻🇳 Vietnamese News Classification - Docker Deployment"
echo "=================================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Stop existing containers
echo "🛑 Stopping existing containers..."
docker-compose down

# Remove old images (optional - uncomment if you want to rebuild completely)
# echo "🗑️  Removing old images..."
# docker rmi vietnamese-news-app_vietnamese-news-app 2>/dev/null || true

# Build and start the application
echo "🏗️  Building and starting the application..."
docker-compose up --build -d

# Wait for the application to start
echo "⏳ Waiting for application to start..."
sleep 10

# Check if the application is running
if curl -f -s http://localhost:5000/ > /dev/null; then
    echo "✅ Application is running successfully!"
    echo "🌐 Access the application at: http://localhost:5000"
    echo "📊 View logs with: docker-compose logs -f"
    echo "🛑 Stop the application with: docker-compose down"
else
    echo "❌ Application failed to start. Check logs with: docker-compose logs"
    docker-compose logs
fi

echo "=================================================="
echo "🔧 Useful Docker commands:"
echo "   View logs: docker-compose logs -f"
echo "   Stop app:  docker-compose down"
echo "   Restart:   docker-compose restart"
echo "   Status:    docker-compose ps"
