FROM python:3.13-slim

WORKDIR /docker_app

# Copy entire project first
COPY . .

# Install dependencies (including -e .)
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]