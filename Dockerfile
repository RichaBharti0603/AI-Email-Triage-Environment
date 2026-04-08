FROM python:3.10-slim

WORKDIR /app

# Copy all project files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the standard port for HF Spaces (7860)
EXPOSE 7860

# Launch the FastAPI app with uvicorn
CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "7860"]