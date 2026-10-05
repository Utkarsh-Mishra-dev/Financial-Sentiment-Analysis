FROM python:3.11-slim

WORKDIR /app

COPY requirements_deploy.txt .
RUN pip install --no-cache-dir -r requirements_deploy.txt

COPY . .

ENV PORT=7860
EXPOSE 7860

CMD ["python", "app.py"]
