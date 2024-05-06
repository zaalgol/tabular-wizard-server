# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory in the container
WORKDIR /tabular-wizard-server

# Copy the current directory contents into the container at /app
COPY . /tabular-wizard-server



# # Install any needed packages specified in requirements.txt
# RUN pip install --trusted-host pypi.python.org -r requirements.txt

# # Install uWSGI
# RUN pip install uwsgi

# Expose port 5000 to be accessible from the outside
EXPOSE 5000

# RUN ls / > tt
# RUN ["sh", "-c", "sleep 10"] 

# RUN ls /app > te
# RUN ["sh", "-c", "sleep 10"] 

# RUN ls /app/config


# Run uWSGI with the ini file
# CMD ["uwsgi", "--ini", "app/config/uwsgi.ini"]
CMD ["sh"] 