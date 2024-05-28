# tabular-wizard-server

`tabular-wizard-server` Runnnig the tabular-wizard server utilizing uWSGI and Docker containers.

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
   git clone https://github.com/zaalgol/tabular-wizard-server

2. **Navigate to the project director**

   ```sh
   cd tabular-wizard-server

3. **Build the Docker image**

   ```sh
   docker build -t tabular-wizard-server .

4. **Download global-bundle.pem file**

5. **Set the correct IP in the .env file. set also the other settings there**

6. **Run the Docker container**

   ```sh
   docker run -p 8080:8080 -v ./.env:/tabular-wizard-server/.env -v ./global-bundle.pem:/tabular-wizard-server/global-bundle.pem tabular-wizard-server
   ```

   Alternatively, if you're using Docker Compose to manage multiple containers:

   ```sh
   docker-compose up
   ````

## Accessing the Application

- **Localhost**: After starting the application, it can be accessed at [http://localhost:8080](http://localhost:8080).
- **Other devices on your network**: Use the host machine's IP address, such as [http://192.168.1.X:8080](http://192.168.1.X:8080).

## Running the Tests

Explain how to run the automated tests for this system. If you have specific test commands or procedures, include them here.

## Deployment

This section would contain additional notes about how to deploy this project on a live system if they differ from the basic installation instructions.

## Built With

- **Flask** - The web framework used.
- **Docker** - Containerization.
- **uWSGI** - Application server for serving Python applications.
- **Docker Compose** - Used for managing multi-container Docker applications.

## Contributing

If you have suggestions for improving the project, please fork the repo and submit a pull request. You can also open an issue with the tag "enhancement". Don't forget to give the project a star! Thanks again!

## Versioning

For the versions available, see the [tags on this repository](https://github.com/zaalgol/flask-uwsgi-docker/tags).

## Authors

- **Zalman Goldstein** - *Initial work* - [ZaalGol](https://github.com/zaalgol)

See also the list of [contributors](https://github.com/zaalgol/flask-uwsgi-docker/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Hat tip to anyone whose code was used as inspiration.
- Thanks to the Flask, Docker, and uWSGI communities for their invaluable resources.

https://www.loginradius.com/blog/engineering/guest-post/securing-flask-api-with-jwt/

