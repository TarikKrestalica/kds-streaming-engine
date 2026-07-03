FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required to compile any source extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all local project code into the container engine
COPY . .

# Expose port 8080 for Cloud Run traffic routing
EXPOSE 8080

# Run Uvicorn ASGI to serve your FastAPI stream app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]