import os
import cv2
import json
from glob import glob
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from src.scripts.random_string_generator import get_random_string_generator


annotation_path = '/home/user/data/experiment/fine_tuning_for_USA/result.json'
data_folder = '/home/user/data/experiment/fine_tuning_for_USA/storage'
output_plates_folder = '/home/user/data/experiment/fine_tuning_for_USA/plates'

all_image_paths = np.array([x for x in glob(os.path.join(data_folder, "**", "*")) if ".jpg" in x])
df = pd.DataFrame(data=all_image_paths, columns=['image_path'])
df["basename"] = df["image_path"].apply(lambda x: os.path.basename(x))

stop = 1

final = []

with open(annotation_path, "r") as f:
    content = json.loads(f.read())

for idx, key in enumerate(content):
    sample = content[key]
    stop = 1
    image_path = df[df["basename"] == key]["image_path"]
    if image_path.shape[0] == 1:
        car_image = cv2.imread(image_path.values[0])

        for jdx, plate in enumerate(sample["objects"]):

            if 'points' not in plate:
                print("ERROR")
                continue

            tl = np.array(plate["points"][1]).astype(int)
            bl = np.array(plate["points"][0]).astype(int)

            tr = np.array(plate["points"][3]).astype(int)
            br = np.array(plate["points"][2]).astype(int)

            w = int(((tr[0] - tl[0]) + (br[0] - bl[0])) / 2)

            h = int(((bl[1] - tl[1]) + (br[1] - tr[1])) / 2)

            plate_coords = np.array([[0, 0], [0, int(h)], [int(w), 0], [int(w), int(h)]], dtype='float32')
            plate_box = np.array([tl, bl, tr, br], dtype='float32')
            transformation_matrix = cv2.getPerspectiveTransform(plate_box, plate_coords)
            lp_img = cv2.warpPerspective(car_image, transformation_matrix, (int(w), int(h)))
            output_dir = Path(
                os.path.join(output_plates_folder, os.path.basename(os.path.dirname(image_path.values[0]))))
            if not output_dir.exists():
                output_dir.mkdir(parents=True, exist_ok=True)
            emirate = plate["class"].lower()
            plate_label = plate["plate_number"]

            if emirate != "error" and plate_label is not None:
                plate_label = plate_label.lower()
                f_name = get_random_string_generator(15, postfix='.jpg').lower()
                cv2.imwrite(os.path.join(output_dir, f_name), lp_img)
                relative_path = os.path.join(os.path.basename(os.path.dirname(output_dir)),
                                             os.path.basename(output_dir), f_name)
                full_data = [relative_path, plate_label, emirate]
                final.append(full_data)
                print(f"Proceeds:{idx}")

df_out = pd.DataFrame(data=final, columns=['image_path', 'plate_label', 'emirate'])
df_out.to_csv('/home/user/data/experiment/fine_tuning_for_USA/annotations.csv', index_label=False, index=False)
