# Use official Python image
FROM python:3.11-slim

# Environment variables
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Create virtual environment
RUN python -m venv $VIRTUAL_ENV

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire app
COPY . .

# Expose the correct port for Hugging Face
EXPOSE 7860

# Run the app using the expected port
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "7860"]
