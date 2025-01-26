# Use Python base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir flask numpy scikit-learn joblib

# Expose port 5000
EXPOSE 5000

# Command to run the app
CMD ["python", "app.py"]
