import cv2
import numpy as np
from ultralytics import YOLO
import sys
import os
from datetime import datetime, timedelta
import time
from send_reports import send_signal, prepare_signal
from upload_image import upload, BUCKET_NAME, IMAGE_PATH, OBJECT_NAME
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

# Получаем аргумент командной строки
if len(sys.argv) < 2:
    print("Не указан источник видео")
    sys.exit(1)

video_source = sys.argv[1]

# Определяем источник видео
if video_source.isdigit():
    video_source = int(video_source)
elif not (video_source.startswith("rtsp://") or video_source.startswith("http://") or os.path.isfile(video_source)):
    print("Неверный формат ввода источника видео")
    sys.exit(1)

# Открываем видеопоток
cap = cv2.VideoCapture(video_source)

# Проверяем, успешно ли открыт видеопоток
if not cap.isOpened():
    print(f"Не удалось открыть источник видео: {video_source}")
    sys.exit(1)

print("Начало обработки видео-потока...")

# Переменные для отслеживания времени и кадров
last_detection_time = datetime.min
frame_counter = 0  # Счётчик кадров
while True:
    violations = [] # reset violations for each new frame
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1

    # Обрабатываем только каждый 10-й кадр
    if frame_counter % 30 != 0:
        continue

    # Преобразуем кадр в RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Используем первую модель для обнаружения людей
    start_time = time.time()
    results = model_for_cut(frame_rgb, classes=[0], verbose=False)
    end_time = time.time()
    print(f"Время работы первой модели: {end_time - start_time:.2f} секунд")

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            min_length = min(x2 - x1, y2 - y1)

            # Пропускаем слишком маленькие объекты
            if min_length < 100:
                continue

            # Вырезаем область с человеком
            person_crop = frame_rgb[y1:y2, x1:x2]

            # Проверяем, что вырезанное изображение валидно
            if person_crop.size == 0:
                continue

            # Изменяем размер вырезанного изображения до 640x640
            person_crop_resized, _ = resize_with_padding_info(person_crop, (640, 640))

            # Сохраняем временный файл
            temp_crop_path = "temp_crop.jpg"
            cv2.imwrite(temp_crop_path, cv2.cvtColor(person_crop_resized, cv2.COLOR_RGB2BGR))

            # Используем вторую модель для предсказания нарушений
            start_time = time.time()
            results_local = model(temp_crop_path, verbose=False, conf=0.518, iou=0.2)
            end_time = time.time()
            print(f"Время работы второй модели: {end_time - start_time:.2f} секунд")

            # Проверяем предсказания
            if len(results_local[0].boxes) > 0:
                for det_box in results_local[0].boxes:
                    class_id = int(det_box.cls[0])
                    class_name = results_local[0].names[class_id]  # Получаем имя класса


                    # Проверяем задержку между обнаружениями
                    current_time = datetime.now()
                    if current_time - last_detection_time >= timedelta(seconds=10):
                        log_violation(class_id, class_name)  # Логируем нарушение
                        # record all detected violations to a list of dictionaries (see prepare_signal)
                        # send each violation as a json file to server
                        send_signal(prepare_signal(violation_type=class_name, bucket_name=BUCKET_NAME, object_name=OBJECT_NAME))

                        last_detection_time = current_time

                        # Добавляем рамку на основной кадр
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Красная рамка
                        cv2.putText(frame, f"{class_name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (0, 0, 255), 2, cv2.LINE_AA)  # Текст над рамкой

                        # Визуализируем кадр с нарушением
                        cv2.imshow("Нарушение", frame)
                        send_frame(frame) # here u can specify the image path, bucket name and object name
                        cv2.waitKey(1000)  # Задержка для отображения кадра


            # Удаляем временный файл
            if os.path.exists(temp_crop_path):
                os.remove(temp_crop_path)


cap.release()
cv2.destroyAllWindows()
print("Обработка завершена. Нарушения записаны в файл 'violations_log.txt'.")