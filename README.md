`flask-uwsgi-docker` is a template for creating a Flask server utilizing uWSGI and Docker containers. This project simplifies the process of deploying Flask applications, making them ready for development and production environments.

## Getting Started

These instructions will help you get a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Before you begin, ensure you have the following installed:

- Docker
- Docker Compose (optional, for managing multi-container Docker applications)

### Installing

Follow these steps to get your development environment running:

1. **Clone the repository**

   ```sh
   git clone https://github.com/zaalgol/flask-uwsgi-docker.git

2. **Navigate to the project director**

   ```sh
   cd flask-uwsgi-docker

3. **Build the Docker image**

   ```sh
   docker build -t flask-uwsgi-docker .

4. **Run the Docker container**

   ```sh
   docker run -p 5000:5000 flask-uwsgi-docke
   ```
   
   Alternatively, if you're using Docker Compose to manage multiple containers:

