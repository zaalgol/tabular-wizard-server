# Dockerfile for FastAPI server

# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Install required system packages for LightGBM
RUN apt-get update && apt-get install -y \
    procps \         
    htop \           
    vim \             
    curl \            
    wget \            
    iputils-ping \    
    net-tools \       
    lsof \            
    libgomp1 \    
    && rm -rf /var/lib/apt/lists/*

# Copy only the required files and folders
COPY app/ /app/app/
COPY requirements.txt /app/
COPY .env /app/
COPY run.py /app/

# Install Python dependencies
RUN pip install -r requirements.txt

# Expose port 8080
EXPOSE 8080

# Run the command to start the server
# CMD ["tail", "-f", "/dev/null"]
CMD ["python", "run.py"]
