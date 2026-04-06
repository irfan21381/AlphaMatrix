# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /code

# Copy requirements first to leverage Docker cache
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port 7860 (Hugging Face default)
EXPOSE 7860

# Command to run the FastAPI backend in the background 
# and the Gradio frontend in the foreground
CMD uvicorn app.main:app --host 0.0.0.0 --port 8000 & python app.py