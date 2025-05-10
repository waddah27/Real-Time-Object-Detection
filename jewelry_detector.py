import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
import os

# Определение модели для классификации украшений
class RegionObjectClassifier(nn.Module):
    def __init__(self, num_region_types=4, pretrained=False):
        super(RegionObjectClassifier, self).__init__()

        # Загрузка ResNet18 (без предобученных весов для инференса)
        self.backbone = torch.hub.load('pytorch/vision', 'resnet18', pretrained=pretrained)

        # Удаляем последний полносвязный слой
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # Слой для обработки one-hot вектора типа области
        self.region_embedding = nn.Linear(num_region_types, 64)

        # Объединенный классификатор
        self.classifier = nn.Sequential(
            nn.Linear(num_features + 64, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, image, region_type):
        # Извлечение признаков из изображения
        img_features = self.backbone(image)

        # Обработка информации о типе области
        region_features = self.region_embedding(region_type)

        # Объединение признаков
        combined_features = torch.cat((img_features, region_features), dim=1)

        # Классификация
        output = self.classifier(combined_features)

        return output

# Глобальные переменные для хранения моделей
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pose_model = None
classifier_model = None

# Функция инициализации моделей
def init_models(classifier_model_path):
    global pose_model, classifier_model, device
    
    print(f"Инициализация моделей на устройстве: {device}")
    
    # Загрузка модели позы
    pose_model = YOLO('models/yolo11x-pose.pt')
    
    # Загрузка классификатора украшений
    classifier_model = RegionObjectClassifier(num_region_types=4, pretrained=False)
    classifier_model.load_state_dict(torch.load(classifier_model_path, map_location=device))
    classifier_model = classifier_model.to(device)
    classifier_model.eval()
    
    print("Модели успешно загружены")

# Функция для предсказания наличия украшения
def predict_object(image, region_type):
    global classifier_model, device
    
    # Трансформации для изображений
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Преобразование изображения
    if isinstance(image, np.ndarray):
        # Если это numpy массив (из OpenCV), преобразуем в PIL Image
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    elif not isinstance(image, Image.Image):
        raise TypeError("Изображение должно быть numpy.ndarray или PIL.Image")

    image_tensor = transform(image).unsqueeze(0).to(device)

    # Маппинг для типов областей
    region_type_to_idx = {'ear': 0, 'hand': 1, 'neck': 2, 'wrist': 3}

    # Проверка и коррекция типа области
    if region_type == 'hear':
        region_type = 'ear'
    elif region_type == 'zapyastie':
        region_type = 'wrist'

    region_type_idx = region_type_to_idx[region_type]
    region_type_onehot = torch.zeros(1, 4).to(device)
    region_type_onehot[0, region_type_idx] = 1.0

    # Пороги уверенности по типу области
    if region_type == 'ear':
        conf = 0.95
    elif region_type == 'wrist':
        conf = 0.95
    elif region_type == 'hand':
        conf = 0.95
    elif region_type == 'neck':
        conf = 0.95
    else:
        conf = 0.9

    # Предсказание
    with torch.no_grad():
        output = classifier_model(image_tensor, region_type_onehot)
        probability = output.item()
        has_object = probability >= conf

    return has_object, probability

# Основная функция для обнаружения украшений
def detect_jewelry(images):
    """
    Функция для обнаружения украшений на изображениях с людьми
    
    Args:
        images: Список изображений с уже вырезанными людьми (список numpy arrays)
        
    Returns:
        Словарь с информацией о наличии украшений: {"кольцо": True/False, "часы": True/False, "бусы": True/False, "серьги": True/False}
    """
    global pose_model
    
    # Проверка инициализации моделей
    if pose_model is None or classifier_model is None:
        raise RuntimeError("Модели не инициализированы. Вызовите init_models() перед использованием detect_jewelry()")
    
    # Маппинг украшений по типам областей
    jewelry_mapping = {
        'ear': 'серьги',
        'hand': 'кольцо',
        'neck': 'бусы',
        'wrist': 'часы'
    }
    
    # Инициализация результирующего словаря
    jewelry_results = {
        "кольцо": False,
        "часы": False,
        "бусы": False,
        "серьги": False
    }
    
    # Обработка каждого изображения
    for person_img in images:
        # Получение ключевых точек позы
        pose_results = pose_model(person_img, verbose=False, stream=False)
        if not pose_results or not hasattr(pose_results[0], 'keypoints'):
            continue

        keypoints = pose_results[0].keypoints.data.cpu().numpy()
        if len(keypoints) == 0:
            continue
        if keypoints.shape[0] == 0:
            continue

        person_keypoints = keypoints[0]
        if person_keypoints.size == 0 or person_keypoints.shape[0] < 11:
            continue
        
        areas = []

        # Определение областей для запястий и рук
        if (sum(person_keypoints[7,:2]) > 0 and sum(person_keypoints[9,:2]) > 0) or (sum(person_keypoints[8,:2]) > 0 and sum(person_keypoints[10,:2]) > 0):
            # Запястья (часы)
            for wrist_idx, elbow_idx, region_type in [(9, 7, 'wrist'), (10, 8, 'wrist')]:
                wx, wy, wc = person_keypoints[wrist_idx]
                ex, ey, ec = person_keypoints[elbow_idx]
                if wc > 0.5 and ec > 0.5:
                    dx, dy = wx - ex, wy - ey
                    dist = (dx**2 + dy**2)**0.5
                    size = dist / 2
                    x1a = int(wx - 0.4 * dist)
                    y1a = int(wy - 0.4 * dist)
                    x2a = int(wx + 0.4 * dist)
                    y2a = int(wy + 0.4 * dist)
                    areas.append((x1a, y1a, x2a, y2a, region_type))

            # Руки (кольца)
            for wrist_idx, elbow_idx, region_type in [(9, 7, 'hand'), (10, 8, 'hand')]:
                wx, wy, wc = person_keypoints[wrist_idx]
                ex, ey, ec = person_keypoints[elbow_idx]
                if wc > 0.5 and ec > 0.5:
                    dx, dy = wx - ex, wy - ey
                    dist = (dx**2 + dy**2)**0.5
                    step = dist / 1.5
                    px = wx + dx / dist * step
                    py = wy + dy / dist * step
                    size = step
                    x1a = int(px - size)
                    y1a = int(py - size)
                    x2a = int(px + size)
                    y2a = int(py + size)
                    areas.append((x1a, y1a, x2a, y2a, region_type))

        # Определение областей для ушей (серьги)
        if sum(person_keypoints[4,:2]) > 0 or sum(person_keypoints[3,:2]) > 0:
            for ear_idx, nose_idx, region_type in [(4, 0, 'ear'), (3, 0, 'ear')]:
                ex, ey, ec = person_keypoints[ear_idx]
                nx, ny, nc = person_keypoints[nose_idx]
                if ec > 0.5 and nc > 0.5:
                    dx, dy = ex - nx, ey - ny
                    dist = (dx**2 + dy**2)**0.5
                    x1a = int(ex - 0.4 * dist)
                    y1a = int(ey - 0.1 * dist)
                    x2a = int(ex + 0.4 * dist)
                    y2a = int(ey + 0.8 * dist)
                    areas.append((x1a, y1a, x2a, y2a, region_type))

        # Определение области для шеи (бусы)
        if (sum(person_keypoints[5,:2]) > 0 and sum(person_keypoints[1,:2]) > 0) or (sum(person_keypoints[6,:2]) > 0 and sum(person_keypoints[2,:2]) > 0):
            if person_keypoints[5,0] < person_keypoints[6,0]:
                continue
            sx, sy, sc = person_keypoints[5]
            sx1, sy1, sc1 = person_keypoints[6]
            g1x, g1y, _ = person_keypoints[1]
            g2x, g2y, _ = person_keypoints[2]
            if (sx + sy) == 0 or (sx1 + sy1) == 0 or (g1x + g1y) == 0 or (g2x + g2y) == 0:
                continue
            if sc > 0.5 and sc1 > 0.5:
                dx, dy = sx - sx1, sy - sy1
                dist = (dx**2 + dy**2)**0.5
                cy = int((sy + sy1) / 2)
                x1a = int(min(sx, sx1) - 0.2 * dist)
                y1a = int(cy - 0.3 * dist)
                x2a = int(max(sx, sx1) + 0.2 * dist)
                y2a = int(cy + dist)
                areas.append((x1a, y1a, x2a, y2a, 'neck'))

        # Обработка каждой области и проверка наличия украшений
        for x1a, y1a, x2a, y2a, region_type in areas:
            x1a = max(0, x1a)
            y1a = max(0, y1a)
            x2a = min(person_img.shape[1], x2a)
            y2a = min(person_img.shape[0], y2a)

            region_img = person_img[y1a:y2a, x1a:x2a]
            if region_img.size == 0:
                continue

            has_object, prob = predict_object(region_img, region_type)
            if has_object:
                jewelry_name = jewelry_mapping[region_type]
                jewelry_results[jewelry_name] = True
    
    return jewelry_results

# Инициализация моделей при импорте модуля