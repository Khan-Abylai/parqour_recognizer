import numpy as np
import torch
import argparse
import torch
import torch.nn as nn
import numpy as np
from src.models.base_model import CRNN, CRNN_2
from src.config import base_config

parser = argparse.ArgumentParser()
parser.add_argument('--weights_path', type=str,
                    default='/data_ssd/wagons/recognizer/weights/wnpr_crnn/127_100_Train:_12.2981,_Accuracy:_0.9866,_Val:_4.7470,_Accuracy:_0.9980,_lr:_1.0000000000000002e-06.pth')
parser.add_argument('--out_path', type=str, default='../../weights/wnpr_recognizer_extd.np')
parser.add_argument('--img_h', type=int, default=64)
args = parser.parse_args()

model = CRNN_2(image_h=base_config.img_h, num_class=base_config.num_class, num_layers=base_config.model_lstm_layers,
             is_lstm_bidirectional=base_config.model_lsrm_is_bidirectional).cuda(0)
model = nn.DataParallel(model)

checkpoint = torch.load(args.weights_path)['state_dict']
model.load_state_dict(checkpoint)
model.eval()
model.cpu()

print(model)


s_dict = model.state_dict()
total = 0
t = 'num_batches_tracked'
np_weights = np.array([], dtype=np.float32)
for k, v in s_dict.items():
    if k[-len(t):] == t:
        continue
    total += v.numel()
    v_reshape = v.reshape(-1)
    np_v = v_reshape.data.numpy()
    np_weights = np.concatenate((np_weights, np_v))

print(total)
print(np_weights.shape)
print(np_weights)
print(np_weights.dtype)

np_weights.tofile(args.out_path)