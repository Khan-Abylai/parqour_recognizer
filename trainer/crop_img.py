import os
import cv2
import pandas as pd
from tqdm import tqdm

def crop_text_regions(image_dir, annotation_file, output_dir):
    """
    Вырезает текстовые области из изображений на основе аннотаций для использования в распознавателе.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Читаем аннотации
    annotations = pd.read_csv(annotation_file)

    for _, row in tqdm(annotations.iterrows(), total=len(annotations)):
        image_path = os.path.join(image_dir, row['filename'])
        output_path = os.path.join(output_dir, row['words'] + "_" + os.path.basename(image_path))

        # Координаты
        xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])

        # Загружаем изображение
        image = cv2.imread(image_path)
        if image is None:
            print(f"Не удалось загрузить изображение: {image_path}")
            continue

        # Вырезаем область текста
        cropped = image[ymin:ymax, xmin:xmax]

        # Сохраняем вырезанное изображение
        cv2.imwrite(output_path, cropped)

def prepare_data_for_recognizer():
    """
    Подготавливает данные для обучения распознавателя текста, используя детектор для вырезания областей.
    """
    # Параметры
    image_dir = "all_data/train/ru_val/"
    annotation_file = "all_data/train/ru_val/labels.csv"
    output_dir = "./all_data/rec_val_crops/"

    print("Начинаем вырезание текстовых областей...")
    crop_text_regions(image_dir, annotation_file, output_dir)
    print("Вырезание завершено!")

if __name__ == "__main__":
    prepare_data_for_recognizer()
