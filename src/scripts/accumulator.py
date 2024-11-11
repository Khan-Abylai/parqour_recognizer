import numpy as np
import pandas as pd

np.random.seed(42)
in_file_path = '/home/user/data/csv/country_from_30_main.csv'
out_file_dir_path ='/home/user/data/experiments/fine_tuning_for_UAE/csv'

chunks = pd.read_csv(in_file_path, chunksize=100000)
df = pd.concat(chunks)

squared = df[df['squared'] == True]

