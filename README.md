# Real-Time-Object-Detection
# Обнаружение объектов в реальном времени

Обнаружение нарушений формы спецодежды в реальном времени с использованием встроенной видеокамеры и обученной модели YOLO v11.

# Проект Real Camera

## Требования

- Python 3.10+

## Установка Python

Если у вас не установлен Python 3.10 или выше, скачайте и установите его с официального сайта: [Python Downloads](https://www.python.org/downloads/).

## Клонирование репозитория

Склонируйте репозиторий на ваш локальный компьютер:

```bash
git clone https://github.com/SergSof/Real-Time-Object-Detection
cd Real-Time-Object-Detection
```

## Установка зависимостей

Создайте виртуальное окружение и активируйте его:

```bash
python -m venv myenv
myenv\scripts\activate
```

Установите необходимые зависимости:

```bash
pip install -r requirements.txt
```

## Запуск основного файла

### 1. Получение видео-потока с внутренней камеры

Для использования внутренней камеры (например, камеры с индексом 0, 1 или 2), выполните следующую команду:

```bash
python main.py 0
```

или

```bash
python main.py 1
```

или

```bash
python main.py 2
```

### 2. Получение видео-потока по IP адресу

Для получения видео-потока по IP адресу, выполните следующую команду:
(замените IP адрес на свой)

```bash
python main.py http://192.168.1.100:8080/video
```

или

```bash
python main.py rtsp://192.168.1.100:8080/video
```

### 3. Получение видео-потока из файла

Для получения видео-потока из файла, укажите в качестве аргумента, путь к видео-файлу, например:

```bash
python main.py C:\Users\Lenovo\OneDrive\Desktop\project_real_camera\video.mp4
```
Если путь содержит кириллицу, пробелы или другие недопустимые символы, необходимо взять в кавычки:
"C:\Users\Lenovo\OneDrive\Рабочий стол\project_real_camera\video.mp4"

**Обработанный видео-файл, будет записан, в корень проекта.**

### 4. Как отправлять обнаруженные нарушения?

Прежде всего, у пользователя должен быть файл `.env` в главной директории, который должен содержать следующие ключи:

```bash
# RABBITMQ_HOST= ...
RABBITMQ_HOST=...
RABBITMQ_PORT_EXPOSE_1=...
RABBITMQ_DEFAULT_USER=...
RABBITMQ_DEFAULT_PASS=...
RABBITMQ_SIGNALS_QUEUE=...


RABBITMQ_HOST_IMG=...
# RABBITMQ_HOST=...
RABBITMQ_PORT_EXPOSE_1_IMG=...
RABBITMQ_DEFAULT_USER_IMG=...
RABBITMQ_DEFAULT_PASS_IMG=...
RABBITMQ_SIGNALS_QUEUE_IMG=...

# Настройки MinIO
MINIO_ROOT_USER=...
MINIO_ROOT_PASSWORD=...
MINIO_API_PORT=...
MINIO_CONSOLE_PORT=...
```

Обнаруженные нарушения отправляются на сервер в `2` форматах:

#### 1. JSON файл:
Который содержит список словарей всех обнаруженных нарушений в кадре, где каждое нарушение представлено в виде словаря, например:

```json
[
  {
    "name": "Нарушение в головном уборе",
    "slug": "headgear",
    "description": "Отсутствие или неправильное использование головного убора"
  },
  {
    "name": "Нарушение в одежде",
    "slug": "clothing",
    "description": "Несоответствие требованиям к одежде"
  }
]
```

Функция, которая используется для отправки JSON формата, называется `send_violations()`. Перед этим нужный формат указывается с помощью функции `prepare_signal()` следующим образом:

```python
def prepare_signal(
        company: str = "ООО 'Пищепром'",
        event_type: str = "violation",   
        employee_name: str = "Иванов Иван Иванович",
        employee_id: str = "55555555-5555-5555-5555-555555555555",
        photo: str = "https://example.com/photo.jpg",
        violation_type: str = "clothing",            
        description: str = "Отсутствие головного убора"
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
```

Алгоритм обнаружения нарушений предоставляет нам `violation_type`.

#### 2. IMAGE файл:
Каждый кадр, содержащий обнаруженные нарушения, сохраняется по умолчанию под именем `test.jpg`. Здесь пользователь может указать `image_path`, `bucket_name` и `object_name`, используя функцию `send_frame()` в `main.py`. Например:

```python
def send_frame(frame, bucket_name: str = BUCKET_NAME, image_path: str = IMAGE_PATH, object_name: str = OBJECT_NAME):
    # Сохраняем кадр с обнаруженными нарушениями
    cv2.imwrite(image_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    # Отправляем его в S3
    upload(bucket_name, image_path, object_name)
```

Эта функция будет вызываться автоматически при обнаружении нарушения в кадре.
``` 

## Примечания

- Убедитесь, что путь к видеофайлу указан правильно и файл доступен по этому пути.
- Для выхода из программы нажмите клавишу 'q' в окне с видео.
- Если вы используете IP-камеру, убедитесь, что камера доступна по указанному адресу.
