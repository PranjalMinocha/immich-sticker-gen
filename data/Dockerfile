# 1. Use a stable Python base
FROM python:3.10-slim

# 2. Prevent Python from buffering logs so you see them in real-time
ENV PYTHONUNBUFFERED=1

# 3. INSTALL SYSTEM DEPENDENCIES
# We bundle Java 21, the 'ps' command, and C compilers/Postgres drivers
RUN apt-get update && apt-get install -y \
    openjdk-21-jre-headless \
    procps \
    gcc \
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 4. CONFIGURE JAVA PATHS
# This is critical for Ubuntu 24.04 so PySpark can find the Java engine
ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# 5. SET WORKING DIRECTORY
WORKDIR /app

# 6. INSTALL PYTHON DEPENDENCIES
# We do this before copying the code to take advantage of Docker caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 7. COPY APPLICATION CODE
# This copies your API, generator, and batch pipeline scripts
COPY . .
