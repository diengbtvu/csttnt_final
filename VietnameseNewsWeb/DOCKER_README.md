# Vietnamese News Classification - Docker Deployment

## Quick Start

### Prerequisites
- Docker Engine 20.10+
- Docker Compose 2.0+

### Deployment Options

#### Option 1: Using Deploy Script (Recommended)
```bash
# Make script executable (if needed)
chmod +x deploy.sh

# Run deployment
./deploy.sh
```

#### Option 2: Manual Docker Compose
```bash
# Build and start
docker-compose up --build -d

# View logs
docker-compose logs -f

# Stop application
docker-compose down
```

#### Option 3: Direct Docker Build
```bash
# Build image
cd VietnameseNewsWeb
docker build -t vietnamese-news-app .

# Run container
docker run -d -p 5000:5000 --name vietnamese-news vietnamese-news-app
```

## Access Application
- **URL**: http://localhost:5000
- **Port**: 5000

## Management Commands

### View Application Status
```bash
docker-compose ps
```

### View Real-time Logs
```bash
docker-compose logs -f
```

### Stop Application
```bash
docker-compose down
```

### Restart Application
```bash
docker-compose restart
```

### Update Application
```bash
# Pull latest changes
git pull

# Rebuild and restart
docker-compose up --build -d
```

## Troubleshooting

### Application Won't Start
1. Check logs: `docker-compose logs`
2. Verify port 5000 is available: `netstat -tlnp | grep 5000`
3. Check Docker daemon: `systemctl status docker`

### Out of Memory
```bash
# Check container resources
docker stats

# Increase Docker memory limit if needed
```

### Permission Issues
```bash
# Fix file permissions
sudo chown -R $USER:$USER .
```

## Production Deployment

### For Server Deployment
1. Update `docker-compose.yml` with your domain
2. Add SSL/TLS certificates
3. Use a reverse proxy (nginx, traefik)
4. Set up log rotation
5. Configure monitoring

### Environment Variables
- `ASPNETCORE_ENVIRONMENT`: Set to `Production`
- `ASPNETCORE_URLS`: Application binding URL
- `LC_ALL` & `LANG`: Vietnamese locale support

## File Structure
```
VietnameseNewsWeb/
├── docker-compose.yml          # Docker Compose configuration
├── deploy.sh                   # Deployment script
├── VietnameseNewsWeb/
│   ├── Dockerfile             # Docker image definition
│   ├── .dockerignore          # Docker ignore file
│   └── [application files]
└── logs/                      # Application logs (created at runtime)
```
