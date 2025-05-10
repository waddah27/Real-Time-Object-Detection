import jewelry_detector
import cv2

def main():
    # Инициализация моделей при первом запуске
    classifier_model_path = 'models/region_object_classifier_result3_94.pth'
    jewelry_detector.init_models(classifier_model_path)

    # Пример списка изображений с людьми
    images = [
        cv2.imread("images/серьги_кольцо_бусы.png"),
    ]
    #/content/серьги_кольцо_бусы.png,
    #/content/часы.png
    #/content/серьги.png

    # Обнаружение украшений
    results = jewelry_detector.detect_jewelry(images)

    print("Результаты обнаружения украшений:")
    print(results)

    # Дальнейшая обработка результатов
    # ...

if __name__ == "__main__":
    main()