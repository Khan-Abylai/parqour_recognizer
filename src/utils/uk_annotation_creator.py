import os
from glob import glob
import pandas as pd
data_folder = '/mnt/data/uk_plates'

folders = glob(os.path.join('/mnt/data/uk_plates', '*'))

annotation = []

for idx, folder in enumerate(folders):
    print(idx, folder)

    content = glob(os.path.join(folder, "*"))
    if len(content) != 0 and len(content) == 3:
        label_path = glob(os.path.join(folder, "*.txt"))[0]
        plate_image_path = glob(os.path.join(folder, "*.png"))[0]
        annotated_plate_image_path = glob(os.path.join(folder, "*.jpg"))[0]

        with open(label_path, 'r') as f:
            label = str(f.read()).strip().lower().replace('\n', '').replace(' ', '').replace('!', '').replace('#',
                                                                                                              '').replace(
                '@', '').replace('?', '').replace('$', '').replace('-', '').replace('.', '').replace('|', '').replace(
                '_', '').replace('=', '').replace('-', '').encode("ascii", "ignore").decode()
        annotation.append([plate_image_path.replace('/mnt/data/', ''), label])
        annotation.append([annotated_plate_image_path.replace('/mnt/data/', ''), label])

df = pd.DataFrame(data=annotation, columns=['image_path','car_labels'])
df.to_csv('/mnt/data/all.csv', index_label=None, index=False)