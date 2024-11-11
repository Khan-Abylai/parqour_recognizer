import albumentations as A
import cv2
import numpy as np
import torch
import glob
from models.base_model import CRNN, CRNN_2
from config import base_config as config
from utils.converter import StrLabelConverter
from dataset.utils import preprocess

if __name__ == '__main__':
    # model = CRNN(image_h=config.img_h, num_class=config.num_class, num_layers=config.model_lstm_layers,
    #                is_lstm_bidirectional=config.model_lsrm_is_bidirectional)

    model = CRNN_2(image_h=config.img_h, num_class=config.num_class, num_layers=config.model_lstm_layers,
                 is_lstm_bidirectional=config.model_lsrm_is_bidirectional)

    model = torch.nn.parallel.DataParallel(model)
    converter = StrLabelConverter(config.alphabet)
    state = torch.load('/data_ssd/wagons/recognizer/weights/wnpr_crnn/127_100_Train:_12.2981,_Accuracy:_0.9866,_Val:_4.7470,_Accuracy:_0.9980,_lr:_1.0000000000000002e-06.pth')

    state_dict = state['state_dict']

    model.load_state_dict(state_dict)
    model.cuda()
    model.eval()
    transformer = A.Compose([A.NoOp()])

    out_path = "../debug/exp3/"

    images = sorted(glob.glob('../img/*'))
    for idx, image in enumerate(images):
        img = cv2.imread(image)
        preprocessed_image = preprocess(img, transform=transformer).unsqueeze(0)

        cuda_image = preprocessed_image.cuda()
        predictions = model(cuda_image)


        predictions = predictions.permute(1, 0, 2).contiguous()
        prediction_size = torch.IntTensor([predictions.size(0)]).repeat(1)
        predicted_probs, predicted_labels = predictions.detach().cpu().max(2)
        # predicted_probs = torch.exp(predicted_probs).permute(1, 0).numpy()
        predicted_probs = np.around(torch.exp(predicted_probs).permute(1, 0).numpy(), decimals=1)
        predicted_test_labels = np.array(converter.decode(predicted_labels, prediction_size, raw=False))
        predicted_raw_test_labels = np.array(converter.decode(predicted_labels, prediction_size, raw=True))

        # cv2.imwrite(out_path + np.array2string(predicted_test_labels).strip("'") + ".jpg", img)
        print(predicted_probs)
        print(image, predicted_test_labels, np.prod(predicted_probs))
