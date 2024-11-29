import easyocr
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch.nn as nn
import os


class BidirectionalLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        try: # multi gpu needs this
            self.rnn.flatten_parameters()
        except: # quantization doesn't work with this
            pass
        recurrent, _ = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        output = self.linear(recurrent)  # batch_size x T x output_size
        return output

class VGG_FeatureExtractor(nn.Module):

    def __init__(self, input_channel, output_channel=256):
        super(VGG_FeatureExtractor, self).__init__()
        self.output_channel = [int(output_channel / 8), int(output_channel / 4),
                               int(output_channel / 2), output_channel]
        self.ConvNet = nn.Sequential(
            nn.Conv2d(input_channel, self.output_channel[0], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(self.output_channel[0], self.output_channel[1], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(self.output_channel[1], self.output_channel[2], 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(self.output_channel[2], self.output_channel[2], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(self.output_channel[2], self.output_channel[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True),
            nn.Conv2d(self.output_channel[3], self.output_channel[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(self.output_channel[3], self.output_channel[3], 2, 1, 0), nn.ReLU(True))

    def forward(self, input):
        return self.ConvNet(input)

class Model(nn.Module):

    def __init__(self, input_channel, output_channel, hidden_size, num_class):
        super(Model, self).__init__()
        """ FeatureExtraction """
        self.FeatureExtraction = VGG_FeatureExtractor(input_channel, output_channel)
        self.FeatureExtraction_output = output_channel
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))

        """ Sequence modeling"""
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(self.FeatureExtraction_output, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, hidden_size))
        self.SequenceModeling_output = hidden_size

        """ Prediction """
        self.Prediction = nn.Linear(self.SequenceModeling_output, num_class)


    def forward(self, input, text):
        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))
        visual_feature = visual_feature.squeeze(3)

        """ Sequence modeling stage """
        contextual_feature = self.SequenceModeling(visual_feature)

        """ Prediction stage """
        prediction = self.Prediction(contextual_feature.contiguous())

        return prediction


# Настройки
input_folder = 'ch4_test_images'  # Укажите путь к папке с изображениями
output_folder = 'rec'  # Папка для сохранения обработанных изображений
font_path = 'DejaVuSans.ttf'  # Укажите путь к шрифту

# Создаем выходную папку, если она не существует
os.makedirs(output_folder, exist_ok=True)

# Создаем reader для распознавания текста
reader = easyocr.Reader(
    ['ru'],
    model_storage_directory='/mnt/d/Project_iman/easy_ocr/EasyOCR/model',
    user_network_directory='/mnt/d/Project_iman/easy_ocr/EasyOCR/user_network',
    recog_network='best_norm_ED'
)

# Загружаем шрифт
font = ImageFont.truetype(font_path, size=12)

# Проходим по всем файлам в папке
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Проверяем расширение файла
        image_path = os.path.join(input_folder, filename)
        print(f"Обрабатывается: {image_path}")

        # Загружаем изображение
        image = cv2.imread(image_path)
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image_pil)

        # Распознаем текст
        results = reader.readtext(image_path)

        # Проходимся по результатам
        for (bbox, text, confidence) in results:
            # Извлекаем координаты прямоугольника
            top_left = tuple(map(int, bbox[0]))
            bottom_right = tuple(map(int, bbox[2]))

            # Проверяем порядок координат
            if top_left[1] > bottom_right[1]:  # Если y0 > y1, меняем их местами
                top_left, bottom_right = (top_left[0], bottom_right[1]), (bottom_right[0], top_left[1])

            if top_left[0] > bottom_right[0]:  # Если x0 > x1, меняем их местами
                top_left, bottom_right = (bottom_right[0], top_left[1]), (top_left[0], bottom_right[1])

            # Рисуем прямоугольник вокруг текста
            draw.rectangle([top_left, bottom_right], outline="green", width=2)

            # Формируем текст для отображения
            label = f"{text} ({confidence:.2f})"

            # Добавляем текст на изображение
            text_position = (top_left[0], top_left[1] - 15 if top_left[1] - 15 > 10 else top_left[1] + 5)
            draw.text(text_position, label, fill="green", font=font)

        # Конвертируем изображение обратно в OpenCV
        image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

        # Сохраняем результат
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, image)
        print(f"Результат сохранен в {output_path}")

print("Обработка завершена.")