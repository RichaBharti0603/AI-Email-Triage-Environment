# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY environment.py .
COPY tasks.py .
COPY reward.py .
COPY grader.py .
COPY agent.py .
COPY test_environment.py .

# Run tests
CMD ["python", "test_environment.py"]