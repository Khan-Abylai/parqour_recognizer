import string

base_folder = '/'

img_w = 160
img_h = 64

img_extensions = ['.jpg', '.png', '.jpeg']
checkpoint_ext = '.pth'
russian_alphabet = 'абвгдежзийклмнопрстуфхцчшщъыьэюя'
alphabet = string.digits + russian_alphabet  # + string.ascii_lowercase
num_class = len(alphabet) + 1

epochs = 100
batch_size = 16  # 128
n_cpu = 4  # 16
lr = 0.0001

model_lstm_layers = 1
model_lsrm_is_bidirectional = True

model_name = 'tmp'
model_extension = '.pth'

checkpoint = ''
checkpoint_dir = ''

data_dir = ''


######
#regions = ["snils"]
#num_regions = len(regions)
