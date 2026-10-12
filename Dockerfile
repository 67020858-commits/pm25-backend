FROM python:3.10-slim

WORKDIR /app

# ติดตั้ง System Dependencies (cron และ curl)
RUN apt-get update && apt-get install -y --no-install-recommends \
    cron \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ติดตั้ง Python Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# คัดลอกซอร์สโค้ดและไฟล์ทั้งหมด
COPY . .

# ตั้งค่า Cron Job ดึงข้อมูลสดอัตโนมัติทุกวัน เวลา 00:05 น.
RUN echo "5 0 * * * root /usr/local/bin/python /app/fetch_and_update_data.py >> /var/log/cron.log 2>&1" > /etc/cron.d/pm25-cron \
    && chmod 0644 /etc/cron.d/pm25-cron \
    && crontab /etc/cron.d/pm25-cron

# สร้างสคริปต์ Entrypoint เพื่อสตาร์ททั้ง Cron Service และ FastAPI Server
RUN echo '#!/bin/sh\nservice cron start\ntouch /var/log/cron.log\nexec uvicorn main:app --host 0.0.0.0 --port 8000' > /app/entrypoint.sh \
    && chmod +x /app/entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/app/entrypoint.sh"]