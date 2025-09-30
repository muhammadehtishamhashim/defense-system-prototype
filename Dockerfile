# Multi-stage Docker build for HifazatAI
# Stage 1: Frontend Builder
FROM node:18-alpine AS frontend-builder

# Set working directory
WORKDIR /app/frontend

# Copy package files first for better caching
COPY frontend/package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy frontend source code
COPY frontend/ ./

# Build frontend
RUN npm run build

# Stage 2: Python Dependencies Builder
FROM python:3.10-slim AS python-builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements file
COPY backend/requirements.prod.txt /tmp/requirements.txt

# Install Python dependencies in virtual environment
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Try to install optional dependencies that might fail
RUN pip install --no-cache-dir cython_bbox || echo "cython_bbox installation failed, continuing without it"

# Stage 3: Production Runtime
FROM python:3.10-slim AS production

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r hifazat && useradd -r -g hifazat hifazat

# Set working directory
WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=python-builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy backend code with proper ownership
COPY --chown=hifazat:hifazat backend/ ./backend/

# Copy built frontend from frontend builder
COPY --from=frontend-builder --chown=hifazat:hifazat /app/frontend/dist ./frontend/dist

# Create necessary directories with proper permissions
RUN mkdir -p /app/data /app/logs /app/models && \
    chown -R hifazat:hifazat /app

# Set environment variables for CPU optimization
ENV PYTHONPATH=/app \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    NUMEXPR_NUM_THREADS=4 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Switch to non-root user
USER hifazat

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start command
CMD ["python", "-m", "uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]