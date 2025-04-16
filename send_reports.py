import pika
import json
import datetime
import os
from dotenv import load_dotenv
# Load environment variables
load_dotenv()

# RabbitMQ settings
RABBITMQ_HOST = os.getenv('RABBITMQ_HOST', 'message-broker')
RABBITMQ_PORT = int(os.getenv('RABBITMQ_PORT_EXPOSE_1', 5672))
RABBITMQ_USER = os.getenv('RABBITMQ_DEFAULT_USER', 'guest')
RABBITMQ_PASS = os.getenv('RABBITMQ_DEFAULT_PASS', 'guest')
RABBITMQ_QUEUE = os.getenv('RABBITMQ_SIGNALS_QUEUE', 'violations_queue')

def prepare_signal(
        company:str = "ООО 'Пищепром'",
        event_type:str = "violation",   
        employee_name:str =  "Иванов Иван Иванович",
        employee_id:str = "55555555-5555-5555-5555-555555555555",
        photo:str = "https://example.com/photo.jpg",
        violation_type:str =  "clothing",            
        description:str = "Отсутствие головного убора"
        ):
        
    signal = {
    "company": company,
    "event_type": event_type,
    "happened_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "employee_name": employee_name,
    "employee_id": employee_id,
    "photo": photo,
    "details": {
        "violation_type": f"{violation_type}",
        "description": f"{description}",
    }
    }
    return signal

def send_violations(violations: list[dict]):
    """Sends a list of violations detected in a single video frame."""
    

    # Connect to RabbitMQ
    credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(
            host=RABBITMQ_HOST,
            port=RABBITMQ_PORT,
            credentials=credentials
        )
    )
    channel = connection.channel()

    # Ensure the queue exists (durable=True for persistence)
    channel.queue_declare(queue=RABBITMQ_QUEUE, durable=True)

    # Publish the message
    channel.basic_publish(
        exchange='',
        routing_key=RABBITMQ_QUEUE,
        body=json.dumps(violations),
        properties=pika.BasicProperties(
            delivery_mode=2,  # Persistent message
        )
    )

    print(f"✅ Frame {frame_id}: Sent {len(violations)} violations")
    connection.close()

if __name__ == "__main__":
    # Example: Simulate violations in a frame
    frame_id = 100  # Example frame number
    violations = [
        {
            "type": "no_helmet",
            "confidence": 0.92,
            "bbox": [100, 150, 200, 250],  # x1, y1, x2, y2
            "employee_id": "emp_123",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
        },
        {
            "type": "no_safety_vest",
            "confidence": 0.87,
            "bbox": [300, 400, 350, 450],
            "employee_id": "emp_456",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }
    ]

    send_violations(frame_id, violations)