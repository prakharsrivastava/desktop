FROM python:3.10-slim

WORKDIR /app

# install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# copy requirements
COPY requirements.txt .

# install python packages
RUN pip install --no-cache-dir -r requirements.txt

# install Jupyter kernel
RUN python -m ipykernel install --user

# working directory
WORKDIR /app/notebooks

# start jupyter
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root", "--no-browser"]