FROM python:3.11

WORKDIR /app

COPY requirements.txt .
RUN apt-get update && apt-get install -y python3-pip
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501


CMD ["streamlit", "run", "main.py", "--server.address=0.0.0.0"]