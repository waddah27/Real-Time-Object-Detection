import cv2
import os
import sys
import time
from datetime import datetime, timedelta
import threading
from queue import Queue

import numpy as np
from ultralytics import YOLO
from send_reports import send_signal, prepare_signal, VIOLATIONS_NAMES
from upload_image import upload, BUCKET_NAME, IMAGE_PATH, OBJECT_NAME
from pprint import pprint

# Путь к моделям
model_for_cut_path = 'models/yolo11l.pt'  # Первая модель для детекции людей
model_path = 'models/NN1_v7_aug.pt'      # Вторая модель для предсказания нарушений
violations = [] # a list to store json file for each detected violation in each frame
# Загружаем модели
model_for_cut = YOLO(model_for_cut_path)
model = YOLO(model_path)

def send_frame(frame, bucket_name:str = BUCKET_NAME, image_path:str = IMAGE_PATH, object_name:str = OBJECT_NAME):
    # Save the frame with detected violations
    cv2.imwrite(image_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    # send it to S3
    upload(bucket_name, image_path, object_name)

# Функция для изменения размера с сохранением пропорций
def resize_with_padding_info(image, target_size):
    target_width, target_height = target_size
    height, width = image.shape[:2]
    aspect_ratio = width / height
    target_aspect_ratio = target_width / target_height

    if aspect_ratio > target_aspect_ratio:
        new_width = target_width
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = target_height
        new_width = int(new_height * aspect_ratio)

    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    padded_image = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2
    padded_image[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_image

    padding_info = {
        'x_offset': x_offset,
        'y_offset': y_offset,
        'new_width': new_width,
        'new_height': new_height,
        'original_width': width,
        'original_height': height
    }

    return padded_image, padding_info

# Функция для записи нарушений в текстовый файл и вывода в терминал
def log_violation(class_id, class_name):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{current_time} / Класс нарушения: {class_id} ({class_name})"

    # Записываем в файл
    with open("violations_log.txt", "a", encoding="utf-8") as log_file:
        log_file.write(log_entry + "\n")

    # Дублируем в терминал
    print(log_entry)


# Configuration for 4 cameras
CAMERAS = [
    {
        'username': 'colpino',
        'password': 'colpino374',
        'ip': '5.17.92.56',
        'port': 554,
        'channel': 1,
        'subtype': 0
    },
    {
        'username': 'colpino',
        'password': 'colpino374',
        'ip': '5.17.92.56',
        'port': 554,
        'channel': 2,
        'subtype': 0
    },
    # Add configurations for cameras 3 and 4
    {
        'username': 'colpino',
        'password': 'colpino374',
        'ip': '5.17.92.56',
        'port': 554,
        'channel': 3,
        'subtype': 0
    },
    {
        'username': 'colpino',
        'password': 'colpino374',
        'ip': '5.17.92.56',
        'port': 554,
        'channel': 4,
        'subtype': 0
    }
]

def build_rtsp_url(camera_config):
    """Build RTSP URL from camera configuration"""
    return (f"rtsp://{camera_config['username']}:{camera_config['password']}@"
            f"{camera_config['ip']}:{camera_config['port']}/cam/realmonitor?"
            f"channel={camera_config['channel']}&subtype={camera_config['subtype']}")

def process_camera_stream(camera_config, camera_id):
    """Process a single camera stream"""
    rtsp_url = build_rtsp_url(camera_config)
    
    # Open video stream
    cap = cv2.VideoCapture(rtsp_url)
    
    if not cap.isOpened():
        print(f"Failed to open camera {camera_id} at {rtsp_url}")
        return
    
    print(f"Started processing stream from camera {camera_id}")
    
    # Variables for tracking time and frames
    last_detection_time = datetime.min
    frame_counter = 0
    
    while True:
        violations = []  # reset violations for each new frame
        ret, frame = cap.read()
        if not ret:
            print(f"Camera {camera_id} stream interrupted. Reconnecting...")
            cap.release()
            cap = cv2.VideoCapture(rtsp_url)
            if not cap.isOpened():
                print(f"Failed to reconnect to camera {camera_id}")
                time.sleep(5)  # Wait before retrying
                continue
            else:
                print(f"Reconnected to camera {camera_id}")
            continue

        frame_counter += 1

        # Process only every 30th frame
        if frame_counter % 30 != 0:
            continue

        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Use first model for person detection
        start_time = time.time()
        results = model_for_cut(frame_rgb, classes=[0], verbose=False)
        end_time = time.time()
        print(f"Camera {camera_id}: First model processing time: {end_time - start_time:.2f} seconds")

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                min_length = min(x2 - x1, y2 - y1)

                # Skip too small objects
                if min_length < 100:
                    continue

                # Crop person area
                person_crop = frame_rgb[y1:y2, x1:x2]

                # Check if crop is valid
                if person_crop.size == 0:
                    continue

                # Resize cropped image to 640x640
                person_crop_resized, _ = resize_with_padding_info(person_crop, (640, 640))

                # Save temporary file
                temp_crop_path = f"temp_crop_cam{camera_id}.jpg"
                cv2.imwrite(temp_crop_path, cv2.cvtColor(person_crop_resized, cv2.COLOR_RGB2BGR))

                # Use second model for violation detection
                start_time = time.time()
                results_local = model(temp_crop_path, verbose=False, conf=0.518, iou=0.2)
                end_time = time.time()
                print(f"Camera {camera_id}: Second model processing time: {end_time - start_time:.2f} seconds")

                # Check predictions
                if len(results_local[0].boxes) > 0:
                    for det_box in results_local[0].boxes:
                        class_id = int(det_box.cls[0])
                        class_name = results_local[0].names[class_id]

                        # Check delay between detections
                        current_time = datetime.now()
                        if current_time - last_detection_time >= timedelta(seconds=10):
                            log_violation(class_id, class_name, camera_id)
                            send_signal(prepare_signal(
                                violation_type=VIOLATIONS_NAMES[class_id]["slug"],
                                description=VIOLATIONS_NAMES[class_id]["description"],
                                bucket_name=BUCKET_NAME,
                                object_name=f"{OBJECT_NAME}_cam{camera_id}"
                            ))

                            last_detection_time = current_time

                            # Add bounding box to frame
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                            cv2.putText(frame, f"{class_name} (Cam {camera_id})", 
                                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                       0.5, (0, 0, 255), 2, cv2.LINE_AA)

                            # Display violation
                            cv2.imshow(f"Violation Camera {camera_id}", frame)
                            send_frame(frame, camera_id)
                            cv2.waitKey(1000)

                # Remove temporary file
                if os.path.exists(temp_crop_path):
                    os.remove(temp_crop_path)

    cap.release()

def log_violation(class_id, class_name, camera_id):
    """Log violation with camera information"""
    with open('violations_log.txt', 'a') as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{timestamp} - Camera {camera_id}: {class_name}\n")

def main():
    # Create and start threads for each camera
    threads = []
    for i, camera_config in enumerate(CAMERAS, 1):
        thread = threading.Thread(
            target=process_camera_stream,
            args=(camera_config, i),
            daemon=True
        )
        thread.start()
        threads.append(thread)
    
    # Wait for all threads to complete (which they won't unless there's an error)
    for thread in threads:
        thread.join()

if __name__ == "__main__":
    main()