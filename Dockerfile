FROM python:3.11-slim

WORKDIR /app

# CPU-only torch keeps the image ~2GB smaller; embedding 200 pages on CPU is fast enough.
RUN pip install --no-cache-dir torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
