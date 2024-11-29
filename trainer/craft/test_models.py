import torch
import cv2
import numpy as np
from model.craft import CRAFT
from utils.util import copyStateDict
from torchvision.transforms import Normalize


def load_model(model_path, device='cuda'):
    """
    Загружает предобученную модель CRAFT.
    """
    model = CRAFT(pretrained=False)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(copyStateDict(state_dict['craft']))
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path, canvas_size=1280, mag_ratio=1.5):
    """
    Подготавливает изображение для модели:
    - Изменяет размер изображения
    - Преобразует его в тензор
    """
    image = cv2.imread(image_path)
    original_size = image.shape[:2]

    # Изменение размера изображения
    target_ratio = canvas_size / max(original_size)
    target_size = (int(original_size[1] * target_ratio), int(original_size[0] * target_ratio))
    resized_image = cv2.resize(image, target_size)
    cv2.imwrite("./res/preprocessed_image.jpg", resized_image)

    # Добавление отступов
    target_canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    target_canvas[:target_size[1], :target_size[0], :] = resized_image
    image = target_canvas

    # Преобразование в тензор
    image = image.astype(np.float32)
    image = (image / 255.0) - np.array([0.485, 0.456, 0.406])
    image = image / np.array([0.229, 0.224, 0.225])
    image = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0)  # NCHW

    print("Input tensor shape:", image.shape)

    return image, original_size

# def postprocess_output(output, original_size, canvas_size=1280):
#     """
#     Постобработка результата детектора:
#     - Перевод результата в исходное разрешение
#     """
#     heatmap = output[0].cpu().detach().numpy()
#     heatmap = cv2.resize(heatmap, (original_size[1], original_size[0]))
#     heatmap = (heatmap * 255).astype(np.uint8)
#     print("Heatmap (min, max):", heatmap.min(), heatmap.max())
#     return heatmap


def postprocess_output(output, original_size, canvas_size=1280):
    """
    Постобработка результата детектора:
    - Перевод результата в исходное разрешение
    """
    heatmap = output.cpu().detach().numpy()

    # Нормализация карты в диапазон [0, 1]
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    # Масштабирование до оригинального размера
    heatmap = cv2.resize(heatmap, (original_size[1], original_size[0]))

    # Преобразование в uint8
    heatmap = (heatmap * 255).astype(np.uint8)
    print("Heatmap (min, max):", heatmap.min(), heatmap.max())
    return heatmap


def test_detector(model, image_path, device='cuda'):
    """
    Тестирует модель детектора на изображении.
    """
    # Предобработка
    image, original_size = preprocess_image(image_path)
    image = image.to(device)

    # Прогон через модель
    with torch.no_grad():
        output, _ = model(image)
        print("Model output shape:", output.shape)

    # Постобработка
    region_map = output[0, :, :, 0]
    affinity_map = output[0, :, :, 1]

    print("Region Map (min, max):", region_map.min().item(), region_map.max().item())
    print("Affinity Map (min, max):", affinity_map.min().item(), affinity_map.max().item())

    region_heatmap = postprocess_output(region_map, original_size)
    affinity_heatmap = postprocess_output(affinity_map, original_size)

    print("Region Heatmap (min, max):", region_heatmap.min(), region_heatmap.max())
    print("Affinity Heatmap (min, max):", affinity_heatmap.min(), affinity_heatmap.max())

    return region_heatmap, affinity_heatmap


def extract_text_boxes(region_map, original_image, threshold=0.7):
    """
    Извлекает bounding boxes (поля текста) из Region Map.
    """
    # Преобразование карты в бинарное изображение
    _, binary_map = cv2.threshold(region_map, int(threshold * 255), 255, cv2.THRESH_BINARY)

    # Поиск контуров
    contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Исходное изображение для визуализации
    output_image = original_image.copy()

    # Прямоугольники вокруг текстовых областей
    boxes = []
    for contour in contours:
        # Строим bounding box
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append((x, y, x + w, y + h))

        # Визуализация прямоугольника
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return output_image, boxes


def extract_text_lines(region_map, affinity_map, original_image, region_threshold=0.5, affinity_threshold=0.3):
    """
    Объединяет Region Map и Affinity Map для выделения целых предложений/строк текста.
    """
    # Преобразование Region Map и Affinity Map в бинарные изображения
    _, binary_region = cv2.threshold(region_map, int(region_threshold * 255), 255, cv2.THRESH_BINARY)
    _, binary_affinity = cv2.threshold(affinity_map, int(affinity_threshold * 255), 255, cv2.THRESH_BINARY)

    # Объединение Region Map и Affinity Map
    combined_map = cv2.bitwise_or(binary_region, binary_affinity)

    # Поиск контуров на объединённой карте
    contours, _ = cv2.findContours(combined_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Исходное изображение для визуализации
    output_image = original_image.copy()

    # Прямоугольники вокруг текстовых областей (строк текста)
    text_lines = []
    for contour in contours:
        # Построение bounding box
        x, y, w, h = cv2.boundingRect(contour)
        text_lines.append((x, y, x + w, y + h))

        # Визуализация прямоугольников
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return output_image, text_lines



if __name__ == "__main__":
    # Путь к модели и изображению
    #model_path = "./CRAFT_clr_amp_29500.pth"  # Укажите путь к вашей модели
    model_path = "./CRAFT_clr_amp_29500.pth"
    image_path = "./data_root_dir/ch4_test_images/cropped_SNILS (14).jpg"  # Укажите путь к изображению EasyOCR/trainer/craft/data_root_dir/ch4_test_images/cropped_SNILS (5).jpg

    # Загрузка модели
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(model_path, device)

    # Тестирование модели
    region_map, affinity_map = test_detector(model, image_path, device)

    # Визуализация результатов
    cv2.imwrite("./res/region_map.jpg", region_map)
    cv2.imwrite("./res/affinity_map.jpg", affinity_map)

    print("Region Map and Affinity Map saved as 'region_map.jpg' and 'affinity_map.jpg'")

    # Открытие оригинального изображения
    original_image = cv2.imread(image_path)

    # Извлечение текстовых областей
    text_boxes_image, boxes = extract_text_boxes(region_map, original_image, threshold=0.7)

    # Сохранение результата
    cv2.imwrite("./res/text_boxes.jpg", text_boxes_image)
    print("Text boxes visualized and saved as 'text_boxes.jpg'")

    # Открытие оригинального изображения
    original_image = cv2.imread(image_path)

    # Извлечение строк текста
    text_lines_image, text_lines = extract_text_lines(region_map, affinity_map, original_image, region_threshold=0.5,
                                                      affinity_threshold=0.3)

    # Сохранение результата
    cv2.imwrite("./res/text_lines.jpg", text_lines_image)
    print(f"Text lines visualized and saved as 'text_lines.jpg'")
    print(f"Detected text lines: {text_lines}")

