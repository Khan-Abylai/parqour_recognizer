import os
import csv
import shutil
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split

# Папка с исходными изображениями и аннотациями
image_folder = "/mnt/d/Project_iman/easy_ocr/EasyOCR/trainer/data/rec"
output_folder = "/mnt/d/Project_iman/easy_ocr/EasyOCR/trainer/all_data"
train_folder = os.path.join(output_folder, "ru_train_filtered")
val_folder = os.path.join(output_folder, "ru_val")

# Создание директорий
os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)

# Получение списка всех XML файлов
xml_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.xml')]

# Разделение данных на обучающие и валидационные
train_xml, val_xml = train_test_split(xml_files, test_size=0.2, random_state=42)

# Функция для обработки данных
def process_data(xml_files, output_folder, csv_file):
    with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filename', 'words', 'xmin', 'ymin', 'xmax', 'ymax'])

        for xml_file in xml_files:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            image_name = root.find('filename').text
            image_path = os.path.join(image_folder, image_name)

            # Копирование изображения в соответствующую папку
            if os.path.exists(image_path):
                shutil.copy(image_path, output_folder)
            else:
                print(f"Изображение {image_name} не найдено, пропускаем...")

            for obj in root.findall('object'):
                word = obj.find('name').text
                bbox = obj.find('bndbox')
                xmin = bbox.find('xmin').text
                ymin = bbox.find('ymin').text
                xmax = bbox.find('xmax').text
                ymax = bbox.find('ymax').text

                writer.writerow([image_name, word, xmin, ymin, xmax, ymax])

# Обработка тренировочных данных
process_data(train_xml, train_folder, os.path.join(train_folder, "labels.csv"))

# Обработка валидационных данных
process_data(val_xml, val_folder, os.path.join(val_folder, "labels.csv"))

print("Данные успешно подготовлены!")
