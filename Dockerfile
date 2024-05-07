# Use an official Python runtime as a parent image
FROM python:3.12

# Set the working directory in the container
WORKDIR /tabular-wizard-server

# Copy the current directory contents into the container at /app
COPY ./app /tabular-wizard-server/app
COPY ./run.py /tabular-wizard-server/run.py
COPY ./requirements.txt /tabular-wizard-server/requirements.txt

# # Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# # Install uWSGI
RUN pip install uwsgi

# Expose port 5000 to be accessible from the outside
EXPOSE 8080

# Run uWSGI with the ini file
CMD ["uwsgi", "--ini", "app/config/uwsgi.ini"]