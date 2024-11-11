import warnings

warnings.filterwarnings("ignore")
import os
import albumentations as A
import cv2
import numpy as np
import torch
import glob
from models.base_model import CRNN
from config import base_config as config
from utils.converter import StrLabelConverter
from dataset.utils import preprocess
from pathlib import Path
import shutil


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
    out_folder = '/home/user/code/rec/'
    model = CRNN(image_h=config.img_h, num_class=config.num_class, num_layers=config.model_lstm_layers,
                 is_lstm_bidirectional=config.model_lsrm_is_bidirectional)
    model = torch.nn.parallel.DataParallel(model)
    converter = StrLabelConverter(config.alphabet)
    state = torch.load('/home/user/code/98_100_Train_9.5728,_Accuracy_0.8624,_Val_6.7491,_Accuracy_0.9537,_lr_1.0000000000000002e-06.pth')
    state_dict = state['state_dict']

    model.load_state_dict(state_dict)
    model.cuda()
    model.eval()
    transformer = A.Compose([A.NoOp()])

    folders = sorted(glob.glob('/mnt/data/uk_square_plates/*'))
    count_of_incorrect = 0
    count_of_correct = 0
    for idx, folder in enumerate(folders):
        plate_image_path = glob.glob(os.path.join(folder, "*.jpg"))
        cropped_image_path = glob.glob(os.path.join(folder, "*.png"))

        label_path = glob.glob(os.path.join(folder, "*.txt"))

        content = glob.glob(os.path.join(folder, "*"))

        if len(content) == 0:
            continue

        if len(content) != 3:
            print('FUCK FUCK')
            print(folder)
        label_path = label_path[0]
        with open(label_path, 'r') as f:
            label = f.read()
            label = str(label).strip().lower().replace('\n', '').replace(' ', '').replace('!', '').replace('#',
                                                                                                           '').replace(
                '@', '').replace('?', '').replace('$', '').replace('-', '').replace('.', '').replace('|', '').replace(
                '_', '').replace('=', '').replace('-', '').encode("ascii", "ignore").decode()

        plate_image_path = plate_image_path[0]
        plate_image_label = predict(model, converter, plate_image_path)

        cropped_image_path = cropped_image_path[0]
        cropped_image_label = predict(model, converter, cropped_image_path)

        if plate_image_label != label and cropped_image_label != label:
            subdir = os.path.dirname(label_path)
            dirname = Path(os.path.join(out_folder, os.path.basename(os.path.dirname(label_path))))
            if not dirname.exists():
                dirname.mkdir()

            shutil.copy(plate_image_path, os.path.join(dirname, os.path.basename(plate_image_path)))
            shutil.copy(cropped_image_path, os.path.join(dirname, os.path.basename(cropped_image_path)))
            print(f'Folder:{folder} is not correct. idx:{idx}')
            print(f'{label} {plate_image_label} {cropped_image_label}')
            count_of_incorrect+=1
        else:
            count_of_correct+=1
    print(f'Overall incorrect:{count_of_incorrect }')