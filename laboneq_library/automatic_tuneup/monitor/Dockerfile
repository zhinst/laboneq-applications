# Use the official Python image as the base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy just the pyproject.toml and the project source code
COPY ../pyproject.toml .
COPY ../tuneup/ ./tuneup/

# Install project dependencies
RUN pip install --no-cache-dir -r pyproject.toml

# Expose the port your FastAPI app listens on
EXPOSE 8000

# Start your application
CMD ["uvicorn", "tuneup.main:app", "--host", "0.0.0.0", "--port", "8000"]
