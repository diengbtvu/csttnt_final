# Build stage
FROM mcr.microsoft.com/dotnet/sdk:7.0 AS build
WORKDIR /src

# Copy project file and restore dependencies
COPY VietnameseNewsWeb.csproj ./
RUN dotnet restore

# Copy all source code
COPY . .

# Build the application
RUN dotnet build -c Release -o /app/build

# Publish the application
RUN dotnet publish -c Release -o /app/publish

# Runtime stage
FROM mcr.microsoft.com/dotnet/aspnet:7.0 AS runtime
WORKDIR /app

# Install necessary dependencies for Vietnamese text processing
RUN apt-get update && apt-get install -y \
    libicu-dev \
    locales \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set up Vietnamese locale
RUN echo "vi_VN.UTF-8 UTF-8" >> /etc/locale.gen && locale-gen

# Copy published app
COPY --from=build /app/publish .

# Copy dataset file
COPY vietnamese_news_dataset_cleaned.csv ./

# Create directory for logs
RUN mkdir -p /app/logs

# Set environment variables
ENV ASPNETCORE_ENVIRONMENT=Production
ENV ASPNETCORE_URLS=http://0.0.0.0:5000
ENV DOTNET_SYSTEM_GLOBALIZATION_INVARIANT=false
ENV LC_ALL=vi_VN.UTF-8
ENV LANG=vi_VN.UTF-8

# Expose port 5000
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/ || exit 1

# Set user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Set the entry point
ENTRYPOINT ["dotnet", "VietnameseNewsWeb.dll"]
