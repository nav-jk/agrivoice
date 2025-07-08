# Use slim Python 3.10 as base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies (ffmpeg for audio)
RUN apt-get update && apt-get install -y ffmpeg && apt-get clean

# Copy requirements and install Python packages
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port (optional but recommended for clarity)
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
