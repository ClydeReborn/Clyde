# Use the official Python 3.12 slim image as the base image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Install git (required for installing some Python packages from Git repositories)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Command to run the application with Python optimization (-OO) and the -r argument
CMD ["python", "main.py"]
