import pandas as pd
import os
from glob import glob
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

prefix = '/data_ssd/wagons/recognizer/'
base_folder = '/data_ssd/wagons/recognizer/images'

images = glob(os.path.join(base_folder,  "*"))
print(f"files:{len(images)}")
data = []

for idx, item in enumerate(images):

    label_file = item.replace("images", "labels").replace(".png", ".txt")

    if os.path.exists(label_file):

        with open(label_file, 'r') as f:
            label = f.read()
        sample = [item.replace(prefix, ''), label]
        data.append(sample)
df = pd.DataFrame(data=np.array(data), columns=['filename', 'labels'])
df.to_csv('/data_ssd/wagons/recognizer/wagon_plates.csv', index_label=False, index=False)


f_path = '/data_ssd/wagons/recognizer/wagon_plates.csv'
df = pd.read_csv(f_path)

train, test = train_test_split(df, test_size=0.1)
stop = 1

print(train)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
# X = df[["filename", "labels"]].values
# y = df["region"].values
# sss.get_n_splits(X, y)
# for train_index, test_index in sss.split(X, y):
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
# y_train = y_train.reshape(-1,1)
# y_test = y_test.reshape(-1,1)
# train = np.concatenate([X_train, y_train], axis=1)
# test = np.concatenate([X_test, y_test], axis=1)
#
df_train = pd.DataFrame(data=train.values, columns=['filename', 'labels'])
df_val = pd.DataFrame(data=test.values, columns=['filename', 'labels'])

print(df_train["filename"].value_counts())
print(df_val["filename"].value_counts())

df_train.to_csv(f_path.replace('wagon_plates.csv', 'train.csv'), index_label=None, index=None)
df_val.to_csv(f_path.replace('wagon_plates.csv', 'test.csv'), index_label=None, index=None)