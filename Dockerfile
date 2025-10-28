FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    libglib2.0-0 \
    libgthread-2.0-0 \
    libfontconfig1 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt ./requirements.txt

# Install Python dependencies with better error handling
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --timeout=1000 -r requirements.txt

# Copy application files
COPY api.py .
COPY best_jitter.pth .
COPY .env .

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8010

# Run the application
CMD ["python", "api.py"]