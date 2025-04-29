FROM python:3.9-slim

WORKDIR /app

# 复制精简版requirements
COPY deployments/flask_app/docker_requirements.txt /app/requirements.txt

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 然后复制其余文件
COPY . /app/

EXPOSE 5000

CMD ["python", "deployments/flask_app/app.py"]