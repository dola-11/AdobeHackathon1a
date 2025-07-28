
FROM --platform=linux/amd64 python:3.11-slim


WORKDIR /app


RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .


RUN pip install --no-cache-dir -r requirements.txt


COPY . .


RUN mkdir -p /app/models
COPY models/*.pkl /app/models/


RUN mkdir -p /app/input /app/output


ENV PYTHONUNBUFFERED=1


CMD ["python", "batch_processor.py"] 
