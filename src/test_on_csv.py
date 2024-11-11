import warnings
warnings.filterwarnings("ignore")
import os
import albumentations as A
import cv2
import numpy as np
import torch
import pandas as pd
from models.base_model import CRNN, CRNN_2
from config import base_config as config
from utils.converter import StrLabelConverter
from dataset.utils import preprocess
from pathlib import Path
import shutil

checkpoint_dir = '/home/user/code'
out_folder = '/home/user/code/rec/'  # path res.
csv_path = '/home/user/code/test.csv'  # path CSV
model_path = '/home/user/code/wnpr_crnn_CRNN2/99_100_Train_24.8658,_Accuracy_0.8283,_Val_8.0170,_Accuracy_0.9193,_lr_1e-05.pth'


def predict(model, converter, image_path):
    img = cv2.imread(image_path)
    preprocessed_image = preprocess(img, transform=transformer).unsqueeze(0)

    cuda_image = preprocessed_image.cuda()
    predictions = model(cuda_image)

    predictions = predictions.permute(1, 0, 2).contiguous()
    prediction_size = torch.IntTensor([predictions.size(0)]).repeat(1)
    predicted_probs, predicted_labels = predictions.detach().cpu().max(2)
    predicted_probs = np.around(torch.exp(predicted_probs).permute(1, 0).numpy(), decimals=1)
    predicted_test_labels = np.array(converter.decode(predicted_labels, prediction_size, raw=False))
    predicted_raw_test_labels = np.array(converter.decode(predicted_labels, prediction_size, raw=True))

    return predicted_test_labels


if __name__ == '__main__':
    correct_folder = os.path.join(out_folder, 'correct')
    incorrect_folder = os.path.join(out_folder, 'incorrect')

    # Создание папок, если их нет
    os.makedirs(correct_folder, exist_ok=True)
    os.makedirs(incorrect_folder, exist_ok=True)

    model = CRNN_2(image_h=config.img_h, num_class=config.num_class, num_layers=config.model_lstm_layers,
                 is_lstm_bidirectional=config.model_lsrm_is_bidirectional)
    model = torch.nn.parallel.DataParallel(model)
    converter = StrLabelConverter(config.alphabet)
    state = torch.load(model_path)
    state_dict = state['state_dict']

    model.load_state_dict(state_dict)
    model.cuda()
    model.eval()
    transformer = A.Compose([A.NoOp()])

    # Чтение данных из CSV
    data = pd.read_csv(csv_path)

    count_of_incorrect = 0
    count_of_correct = 0

    for idx, row in data.iterrows():
        image_path = checkpoint_dir + '/' + row['filename']
        label = str(row['labels']).strip().lower().replace('\n', '').replace(' ', '').replace('!', '').replace('#', '').replace(
            '@', '').replace('?', '').replace('$', '').replace('-', '').replace('.', '').replace('|', '').replace(
            '_', '').replace('=', '').replace('-', '')  #.encode("ascii", "ignore").decode()

        # Предсказания модели
        image_label = predict(model, converter, image_path)

        if image_label != label:
            shutil.copy(image_path, os.path.join(incorrect_folder,
                                                 os.path.basename(image_path)[:-4] + "_" + str(image_label) + os.path.basename(
                                                     image_path)[-4:]))
            print(f'Image:{image_path} is not correct. idx:{idx}')
            print(f'{label} {image_label}')
            count_of_incorrect += 1
        else:
            shutil.copy(image_path, os.path.join(correct_folder, os.path.basename(image_path)))
            count_of_correct += 1

    print(f'Overall correct: {count_of_correct}')
    print(f'Overall incorrect: {count_of_incorrect}')

