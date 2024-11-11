import torch
from models.base_model import CRNN
from config import base_config as config
from utils.converter import StrLabelConverter

checkpoint_path = '/home/user/recognizer_pipeline/weights/recognizer_base.pth'

model = CRNN(image_h=config.img_h, num_class=config.num_class, num_layers=config.model_lstm_layers,
             is_lstm_bidirectional=config.model_lsrm_is_bidirectional)
model = torch.nn.parallel.DataParallel(model)
converter = StrLabelConverter(config.alphabet)
state = torch.load(checkpoint_path)
state_dict = state['state_dict']

model.load_state_dict(state_dict)
model.cuda()
model.eval()

if isinstance(model, torch.nn.DataParallel):
    model = model.module

model.eval()
model.cpu()
dummy_input = torch.randn(1, 3, 32, 128).cpu()
input_names = ["actual_input"]
output_names = ["output"]

torch.onnx.export(model, dummy_input, checkpoint_path.replace('.pth', '.onnx'), verbose=True, input_names=input_names,
                  output_names=output_names, export_params=True, opset_version=11)
print("Model converted to the onnx")
