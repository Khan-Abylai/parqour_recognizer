import os
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

random.seed(42)
np.random.seed(42)
in_file_path = '/home/user/data/csv/country_from_30_main.csv'

chunks = pd.read_csv(in_file_path, chunksize=100000)
base_df = pd.concat(chunks)

base_df.drop(
    base_df[base_df['squared']==True].index, inplace=True,
)

base_df = base_df.reset_index(drop=True)

squared_df = pd.read_csv('/home/user/data/experiment/fine_tuning_for_UAE/csv/squared.csv')


final = pd.concat(
    [base_df, squared_df], axis=0
)

main = final[['filename', 'label']]
main.columns = ['image_path', 'car_labels']

train_data, val_data = train_test_split(main, test_size=0.1, random_state=42)


train_data = train_data.reset_index(drop=True)
val_data = val_data.reset_index(drop=True)

train_data.to_csv(os.path.join('/home/user/data/experiment/fine_tuning_for_UAE/csv', 'train.csv'), index=False, index_label=False)
val_data.to_csv(os.path.join('/home/user/data/experiment/fine_tuning_for_UAE/csv', 'val.csv'), index=False, index_label=False)




# squared = base_df[base_df['squared'] == True].reset_index(drop=True)
# is_even = squared.index % 2 == 0
# even = squared[is_even]
# odd = squared[~is_even]
# odd.columns = [x + "2" for x in odd.columns]
#
# even = even.reset_index(drop=True)
# odd = odd.reset_index(drop=True)
# even['index'] = even.index
# odd['index'] = odd.index
# stop = 1
# df = pd.merge(even, odd, on='index')
#
# data_dir = '/home/user/data'
# debug_dir = '/home/user/parking_recognizer/debug'
# out_dir = '/home/user/data/data/square_concat'
#
# path_length = 10
# samples = []
# for idx, row in df.iterrows():
#     f_name_1 = row['filename']
#     f_name_2 = row['filename2']
#
#     label_1 = row['label']
#     label_2 = row['label2']
#
#     top_half1 = row['top_half']
#     top_half2 = row['top_half2']
#
#     if top_half2 and top_half1:
#         continue
#
#     t_full_path = os.path.join(data_dir, f_name_1)
#     b_full_path = os.path.join(data_dir, f_name_2)
#
#     t_out_path = os.path.join(debug_dir, get_random_string_generator(10, postfix='.jpg'))
#     b_out_path = os.path.join(debug_dir, get_random_string_generator(10, postfix='.jpg'))
#
#     top = cv2.imread(t_full_path)
#     bottom = cv2.imread(b_full_path)
#
#     white_image = np.ones((32, 128, 3), dtype=np.uint8) * np.random.randint(0, 255, (3), dtype=np.uint8)
#
#     image = cv2.resize(top, (64, 64))
#
#     white_image[:, 0:64, :] = image[:32, :, :]
#     white_image[:, 64:128, :] = image[32:, :, :]
#
#     # cv2.imwrite(os.path.join(debug_dir, get_random_string_generator(10, postfix='.jpg')), white_image)
#     final_label = label_1 + label_2
#
#     out_path = os.path.join(out_dir, get_random_string_generator(15, postfix='.jpg').lower())
#     cv2.imwrite(out_path, white_image)
#
#
#     data = [out_path.replace(data_dir+'/', ''), final_label, 'kz', False, False]
#     samples.append(data)
# columns = ['filename', 'label', 'country', 'squared', 'top_half']
# df = pd.DataFrame(data=samples, columns=columns)
# df.to_csv('/home/user/data/experiment/fine_tuning_for_UAE/csv/squared.csv', index_label=False, index=False)