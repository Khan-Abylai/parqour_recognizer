import albumentations as A

try:
    from src.augmentation.transforms import transform_old
except:
    from augmentation.transforms import transform_old

train_csv = ['/home/user/code/train.csv']

val_csv = ['/home/user/code/test.csv']

params = {
    'train': {
        'transform': transform_old,
        'train': True,
        'csv_files': train_csv
    },
    'val': {
        'transform': A.Compose([A.NoOp()]),
        'train': False,
        'csv_files': val_csv
    }
}
