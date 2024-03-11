# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Install uWSGI
RUN pip install uwsgi

# Expose port 8080 to be accessible from the outside
EXPOSE 8080

# Run uWSGI with the ini file
CMD ["uwsgi", "--ini", "config/uwsgi.ini"]
