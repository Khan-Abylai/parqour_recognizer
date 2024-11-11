import os
from glob import glob
import shutil
from pathlib import Path

image_base_folder = '/home/user/data/data'
correct_folder = '/home/user/data/correct'

image_list = glob(os.path.join(image_base_folder, '**', '**', "*"))

image_list_dict = {}
correct_image_list_dict = {}

for item in image_list:
    key = item.replace(image_base_folder, '')
    image_list_dict[key] = item

correct_image_list = glob(os.path.join(correct_folder, '**', "**", "*"))
for item in correct_image_list:
    key = item.replace(correct_folder, '')
    correct_image_list_dict[key] = item
index=0
for idx, folder in enumerate(image_list):
    content = glob(os.path.join(folder, "*"))

    is_square = any([True if 'square' in x else False for x in content])
    plate_label_file = [x for x in content if '.txt' in x][0]
    synthetic_image = [x for x in content if 'synthetic' in x][0]
    real_image = [x for x in content if (is_square and 'concat' in x) or (not is_square and 'real' in x)]
    if len(real_image) == 1:
        key = folder.replace(image_base_folder, '')
        if key not in correct_image_list_dict:
            new_folder = Path(image_list_dict[key].replace(image_base_folder, correct_folder))
            if not new_folder.exists():
                new_folder.mkdir(parents=True, exist_ok=True)
            dst_synthetic_image_path = os.path.join(new_folder, 'synthetic.jpg')
            shutil.copy(synthetic_image, dst_synthetic_image_path)
            shutil.copy(plate_label_file, os.path.join(new_folder,'plate.txt'))
            print(idx)
            index+=1
print(index)