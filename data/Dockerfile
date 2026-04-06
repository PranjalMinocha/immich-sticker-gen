# Use a lightweight Python base image
FROM python:3.10-slim

# Prevent Python from buffering stdout/stderr to ensure real-time logs in Docker
ENV PYTHONUNBUFFERED=1

# Install Java, procps (for 'ps' command), and DB drivers in one go
RUN apt-get update && apt-get install -y \
    openjdk-17-jre-headless \
    procps \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the Java Home path so Spark can find its engine
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required for compiling Python database drivers
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .