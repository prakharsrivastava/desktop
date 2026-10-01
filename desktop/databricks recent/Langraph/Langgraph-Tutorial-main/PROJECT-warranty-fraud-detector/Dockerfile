FROM python:3.11-slim

WORKDIR /app

#Copy project files
COPY . /app

# Upgrade pip and install dependencies. The repository currently has a
RUN pip install --upgrade pip \
    && sh -c "pip install -r requirements.txt"

# Expose Streamlit default port
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "server.address= 0.0.0.0"]