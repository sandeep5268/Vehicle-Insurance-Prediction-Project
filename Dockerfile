# An Official python 3.12 image from Docker Hub
FROM python:3.12

# Set the working directory
WORKDIR /app

# Copy application code
COPY . /app

# Install the dependencies
RUN pip install -r requirements.txt

# Expose the port FastAPi will run on
EXPOSE 5000

# Command to run the FastAPI app
CMD ["python3", "app.py"]