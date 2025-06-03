# Use the official Python slim image as the base
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements_deploy.txt ./
RUN pip install --no-cache-dir -r requirements_deploy.txt

# Copy the application code
COPY vector_search_app.py ./

# Set Streamlit's configuration to bind to 0.0.0.0 and use port 8080, which is required by Cloud Run
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLE_CORS=false
ENV STREAMLIT_SERVER_PORT=8080

# Expose port 8080
EXPOSE 8080

# Run the Streamlit app
CMD ["streamlit", "run", "vector_search_app.py", "--server.port=8080", "--server.address=0.0.0.0"] 