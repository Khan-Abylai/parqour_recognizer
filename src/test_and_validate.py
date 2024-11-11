import os.path
import shutil
from Levenshtein import distance as lev

import albumentations as A
import cv2
import numpy as np
import torch
import glob
from pathlib import Path
from models.base_model import CRNN
from config import base_config as config
from utils.converter import StrLabelConverter
from dataset.utils import preprocess

model = CRNN(image_h=config.img_h, num_class=config.num_class, num_layers=config.model_lstm_layers,
             is_lstm_bidirectional=config.model_lsrm_is_bidirectional)
model = torch.nn.parallel.DataParallel(model)
converter = StrLabelConverter(config.alphabet)
state = torch.load('/home/user/recognizer_pipeline/weights/recognizer_base.pth')
state_dict = state['state_dict']

model.load_state_dict(state_dict)
model.cuda()
model.eval()
transformer = A.Compose([A.NoOp()])

if __name__ == '__main__':
    image_base_folder = '/home/user/data/data'
    image_list = glob.glob(os.path.join(image_base_folder, '**', '**', "*"))

    correct_folder = '/home/user/data/correct'
    incorrect_folder = '/home/user/data/incorrect'

    for idx, folder in enumerate(image_list):
        print('-' * 100)
        print(f'Working with idx:{idx}, folder;{folder}')
        content = glob.glob(os.path.join(folder, "*"))

        is_square = any([True if 'square' in x else False for x in content])
        plate_label_file = [x for x in content if '.txt' in x][0]
        synthetic_image = [x for x in content if 'synthetic' in x][0]
        real_image = [x for x in content if (is_square and 'concat' in x) or (not is_square and 'real' in x)]

        if len(real_image) == 0:
            dir = Path(os.path.dirname(synthetic_image).replace(image_base_folder, correct_folder))
            if not dir.exists():
                dir.mkdir(parents=True, exist_ok=True)

            shutil.copy(synthetic_image, os.path.join(dir, 'synthetic.jpg'))
            shutil.copy(plate_label_file, os.path.join(dir, 'plate.txt'))
            print(f"There is only synthetic image. folder:{folder}")
        else:
            real_image = real_image[0]
            img = cv2.imread(real_image)
            preprocessed_image = preprocess(img, transform=transformer).unsqueeze(0)

            cuda_image = preprocessed_image.cuda()
            predictions = model(cuda_image)

            predictions = predictions.permute(1, 0, 2).contiguous()
            prediction_size = torch.IntTensor([predictions.size(0)]).repeat(1)
            predicted_probs, predicted_labels = predictions.detach().cpu().max(2)
            predicted_probs = np.around(torch.exp(predicted_probs).permute(1, 0).numpy(), decimals=1)
            predicted_test_labels = np.array(converter.decode(predicted_labels, prediction_size, raw=False))
            predicted_raw_test_labels = np.array(converter.decode(predicted_labels, prediction_size, raw=True))

            pred_label = predicted_test_labels.item()

            with open(plate_label_file, 'r') as f:
                plate_label = f.read()
            whitelist = set('abcdefghijklmnopqrstuvwxyz1234567890')
            label = str(plate_label).strip().lower().replace('\n', '').replace(' ', '').replace('!', '').replace('#',
                                                                                                                 '').replace(
                '@', '').replace('?', '').replace('$', '').replace('-', '').replace('.', '').replace('|', '').replace(
                '_', '').replace('`', '').replace('=', '').encode("ascii", "ignore").decode()
            real_label = ''.join(filter(whitelist.__contains__, label))

            dist = lev(pred_label, real_label)

            if dist > 1:
                dir = Path(os.path.dirname(synthetic_image).replace(image_base_folder, incorrect_folder))
                if not dir.exists():
                    dir.mkdir(parents=True, exist_ok=True)
                shutil.copy(synthetic_image, os.path.join(dir, 'synthetic.jpg'))
                shutil.copy(plate_label_file, os.path.join(dir, 'plate.txt'))
                shutil.copy(real_image, os.path.join(dir, 'real.jpg'))
                print(f'At folder:{folder} distance more than 1. Real:{real_label} pred_label:{pred_label}')
            else:
                dir = Path(os.path.dirname(synthetic_image).replace(image_base_folder, correct_folder))
                if not dir.exists():
                    dir.mkdir(parents=True, exist_ok=True)
                shutil.copy(synthetic_image, os.path.join(dir, 'synthetic.jpg'))
                shutil.copy(plate_label_file, os.path.join(dir, 'plate.txt'))
                shutil.copy(real_image, os.path.join(dir, 'real.jpg'))
                print(f'At folder:{folder} distance less than 1 or equal. Real:{real_label} pred_label:{pred_label}')
