FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files to root of /app
COPY . .

# Launch inference automatically with default flags
CMD ["python", "inference.py", "--difficulty", "easy", "--episodes", "1"]