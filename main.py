import cv2
import numpy as np
from ultralytics import YOLO
import sys
import os
import time
import subprocess
import threading

# Функция для предикта и получения ббоксов
def get_bbox(image, model, colors):
    if image is None or image.size == 0:
        return None
    
    try:
        results = model(image, 
                        save=False, 
                        imgsz=640, 
                        conf=0.4,
                        iou=0.3,
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
    except Exception as e:
        print(f"Ошибка при обработке кадра: {e}")
        return image

# Класс для работы с буферизированным видеопотоком
class RTSPVideoStream:
    def __init__(self, src):
        self.src = src
        self.stopped = False
        self.frame = None
        self.connection_attempts = 0
        self.max_attempts = 5
        self.reconnect()
        
        # Запускаем поток для чтения кадров
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
    
    def reconnect(self):
        if self.connection_attempts >= self.max_attempts:
            print("Превышено количество попыток подключения")
            self.stopped = True
            return False
            
        # Устанавливаем параметры для FFMPEG
        if isinstance(self.src, str) and self.src.startswith("rtsp://"):
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|timeout;12000000"
            self.stream = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
            
            # Устанавливаем буфер
            self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 3)
            
            # Если разрешение слишком высокое, снижаем его
            width = self.stream.get(cv2.CAP_PROP_FRAME_WIDTH)
            if width > 1280:
                self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        else:
            self.stream = cv2.VideoCapture(self.src)
            
        self.connection_attempts += 1
        
        if not self.stream.isOpened():
            print(f"Не удалось подключиться (попытка {self.connection_attempts})")
            time.sleep(2)  # Пауза перед повторной попыткой
            return self.reconnect()
            
        print(f"Подключение успешно (попытка {self.connection_attempts})")
        self.connection_attempts = 0
        return True
    
    def update(self):
        while not self.stopped:
            if not self.stream.isOpened():
                if not self.reconnect():
                    break
                continue
                
            # Пропускаем несколько кадров для очистки буфера
            for _ in range(2):
                self.stream.grab()
                
            # Читаем кадр
            ret, frame = self.stream.read()
            
            if not ret:
                print("Ошибка при чтении кадра, попытка переподключения...")
                if not self.reconnect():
                    break
                continue
                
            self.frame = frame
            
            # Добавляем небольшую задержку, чтобы снизить нагрузку на CPU
            time.sleep(0.01)
    
    def read(self):
        return self.frame
    
    def get(self, prop):
        return self.stream.get(prop)
    
    def stop(self):
        self.stopped = True
        if self.stream.isOpened():
            self.stream.release()

def main():
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
    
    # Используем буферизированный видеопоток для RTSP
    if isinstance(video_source, str) and video_source.startswith("rtsp://"):
        vs = RTSPVideoStream(video_source)
        time.sleep(2.0)  # Позволяем камере инициализироваться
        
        # Получаем параметры видео
        frame_width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vs.get(cv2.CAP_PROP_FPS)
    else:
        # Для обычных видеофайлов используем стандартный подход
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
    
    frame_count = 0
    last_time = time.time()
    fps_counter = 0
    
    try:
        while True:
            # Читаем кадр из соответствующего источника
            if isinstance(video_source, str) and video_source.startswith("rtsp://"):
                frame = vs.read()
                if frame is None:
                    continue
            else:
                ret, frame = cap.read()
                if not ret:
                    break
            
            # Увеличиваем счетчик кадров и вычисляем FPS
            fps_counter += 1
            if time.time() - last_time >= 5.0:
                fps_real = fps_counter / (time.time() - last_time)
                print(f"Текущий FPS: {fps_real:.2f}")
                fps_counter = 0
                last_time = time.time()
            
            # Предикт - получаем кадр с б-боксами
            processed_frame = get_bbox(frame, model, colors)
            
            if processed_frame is None:
                continue
            
            if record_video:  # Записываем кадр в выходное видео, если нужно вести запись
                out.write(processed_frame)
            else:
                cv2.imshow('Real-Time Object Detection', processed_frame)  # Отображаем кадр в окне
            
            frame_count += 1
            
            # Выход при нажатии клавиши 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("Прервано пользователем")
    except Exception as e:
        print(f"Произошла ошибка: {e}")
    finally:
        # Освобождаем ресурсы
        if isinstance(video_source, str) and video_source.startswith("rtsp://"):
            vs.stop()
        else:
            cap.release()
            
        if record_video:
            out.release()
            print("Файл записан: 'result_video.mp4'")
            
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
