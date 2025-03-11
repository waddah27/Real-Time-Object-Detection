import cv2
import numpy as np
from ultralytics import YOLO
import sys
import os

#функция для предикта и получения ббоксов
def get_bbox(image, model, colors):
    results = model(image, 
                    save=False, 
                    imgsz=640, 
                    conf=0.4,
                    verbose=False)    

    # Обрабатываем результаты
    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls)  # Получаем ID класса
            # Извлекаем координаты как массив
            xyxy = box.xyxy.cpu().numpy()[0]  # Преобразуем в одномерный массив
            x1, y1, x2, y2 = map(int, xyxy)  # Преобразуем координаты в целые числа
            color = colors.get(class_id, (255, 255, 255))  # Цвет по классу, белый по умолчанию
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)  # Рисуем рамку
            # Отображаем текст класса
            class_name = model.names[class_id]  # Получаем название класса из модели
            cv2.putText(image, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 
                        0.5, color, 2, cv2.LINE_AA)  # Добавляем текст над рамкой
    return image

# Путь к модели
model_path = 'fartuk_P92-R85_B_s1_v11.pt'
model = YOLO(model_path)

# Определяем цвета для классов
colors = {
    0: (255, 0, 0),    # Синий (спецодежда)
    1: (0, 255, 0),    # Зеленый (головной убор)
    2: (0, 165, 255),  # Оранжевый (нет спецодежды)
    3: (0, 0, 255)     # Красный (нет головного убора)
}

# Получаем аргумент командной строки
if len(sys.argv) < 2:
    print("Не указан источник видео")
    sys.exit(1)

video_source = sys.argv[1]
record_video = False

# Определяем источник видео
if video_source.isdigit():
    video_source = int(video_source)
elif video_source.startswith("rtsp://"):
    pass  # RTSP-адрес, оставляем как есть
elif video_source.count('.') == 3:
    pass  # IP-адрес, оставляем как есть
elif os.path.isfile(video_source):
    record_video = True    
else:
    print("Неверный формат ввода источника видео")
    sys.exit(1)

# Открываем видеопоток
cap = cv2.VideoCapture(video_source)

# Получаем параметры видео
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Разрешение картинки: {frame_width}x{frame_height}, FPS: {fps}")
print("Для выхода нажмите 'q'")

# Определяем кодек и создаем объект VideoWriter, если нужно вести запись
if record_video:
    out = cv2.VideoWriter('result_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    print("Запись видео...")

n = 0
while True:
    # Читаем кадр
    ret, frame = cap.read()
            
    # Если кадр не был прочитан, выходим из цикла
    if not ret:
        break

    # Предикт - получаем кадр с б-боксами
    frame = get_bbox(frame, model, colors)
    
    
    if record_video: # Записываем кадр в выходное видео, если нужно вести запись
        out.write(frame)
    else:
        cv2.imshow('Real-Time Object Detection', frame) # Отображаем кадр в окне

    n += 1
    # Выход при нажатии клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождаем ресурсы
cap.release()
if record_video:
    out.release()
    print("Файл записан: 'result_video.mp4'")
cv2.destroyAllWindows()