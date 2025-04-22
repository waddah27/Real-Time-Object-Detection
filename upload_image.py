import os
from minio import Minio
from dotenv import load_dotenv
from datetime import timedelta

# Загружаем переменные окружения
load_dotenv()

# Конфигурация MinIO
MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT', 'localhost:9000')
MINIO_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY', 'minioadmin')
MINIO_SECRET_KEY = os.getenv('MINIO_SECRET_KEY', 'minioadmin')
MINIO_SECURE = os.getenv('MINIO_SECURE', 'false').lower() == 'true'

# Инициализация клиента MinIO
client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=MINIO_SECURE
)

# Параметры загрузки
BUCKET_NAME = 'images'
IMAGE_PATH = 'test.png'
OBJECT_NAME = 'test.png'

def upload(bucket_name=BUCKET_NAME, image_path=IMAGE_PATH, object_name=OBJECT_NAME):
    # Создаем бакет, если его нет
    found = client.bucket_exists(bucket_name)
    if not found:
        client.make_bucket(bucket_name)
        print(f"Created bucket {bucket_name}")

    # Загружаем файл
    client.fput_object(
        bucket_name,
        object_name,
        image_path,
    )
    print(f"Successfully uploaded {image_path} to bucket {bucket_name}")

    # Получаем URL для доступа к файлу
    url = f"http://storage:9000/{bucket_name}/{object_name}"
    print(f"\nURL to access the image: {url}")

if __name__ == "__main__":
    upload() 