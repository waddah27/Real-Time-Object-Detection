import pika
import json
import datetime
import uuid
import os

# Параметры подключения к RabbitMQ
RABBITMQ_HOST="kolpino-control-api.taskcomplete.ru"
RABBITMQ_PORT=5672
RABBITMQ_USER="rmqser"
RABBITMQ_PASS="pCQYpvzywiTKs19xFk"
RABBITMQ_QUEUE="signals_queue"

RABBITMQ_QUEUE = os.getenv('RABBITMQ_SIGNALS_QUEUE', 'signals_queue')


def send_signal():
    # Создаем тестовый сигнал
    signal = {
        "company": "ООО 'Пищепром'",
        "event_type": "violation",
        "happened_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "employee_name": "Иванов Иван Иванович",
        "employee_id": "55555555-5555-5555-5555-555555555555",
        "photo": "http://localhost:9000/images/test.jpg",
        "details": {
            "violation_type": "headgear",
            "description": "Отсутствие головного убора",
        }
    }

    # Подключаемся к RabbitMQ
    credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(
            host=RABBITMQ_HOST,
            port=RABBITMQ_PORT,
            credentials=credentials
        )
    )
    channel = connection.channel()

    # Объявляем очередь
    channel.queue_declare(queue=RABBITMQ_QUEUE, durable=True)

    # Отправляем сообщение
    channel.basic_publish(
        exchange='',
        routing_key=RABBITMQ_QUEUE,
        body=json.dumps(signal),
        properties=pika.BasicProperties(
            delivery_mode=2,  # make message persistent
        )
    )

    print(f"Отправлен сигнал: {json.dumps(signal, indent=2, ensure_ascii=False)}")
    connection.close()