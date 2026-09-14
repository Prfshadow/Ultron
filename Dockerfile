# Ultron Dockerfile
# Build: docker build -t ultron .
# Run: docker run -d -p 5000:5000 --env-file .env -v ultron-data:/app/data -v ultron-logs:/app/logs ultron

FROM python:3.12-slim

# Install system dependencies for PDF, OCR, and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    libtesseract-dev \
    libmagic1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r ultron && useradd -r -g ultron ultron

# Set working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories and set permissions
RUN mkdir -p /app/data /app/logs /app/uploads /app/data/kb_inbox/_done && \
    chown -R ultron:ultron /app

# Switch to non-root user
USER ultron

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:5000/ || exit 1

# Production server (waitress is already in requirements.txt).
# Honors FLASK_HOST / FLASK_PORT from the environment (see docker-compose.yml).
CMD waitress-serve --listen=${FLASK_HOST:-0.0.0.0}:${FLASK_PORT:-5000} --threads=4 app:app