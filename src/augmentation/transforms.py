
import warnings
warnings.filterwarnings("ignore")

import albumentations as A

try:
    from src.augmentation.aug_mix import RandomAugMix
except:
    from augmentation.aug_mix import RandomAugMix
from albumentations import (
    CLAHE, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, OneOf, Compose, ChannelDropout,
    InvertImg, ChannelShuffle, MultiplicativeNoise, JpegCompression
)

transform_album = A.Compose(
    [
        A.Rotate(limit=45),
        A.HorizontalFlip(),
        A.Blur(),
        A.GaussNoise(),
        A.RandomShadow(),
        A.Cutout(p=0.3, num_holes=16, max_h_size=16, max_w_size=16),
    ]
)

transform_album_hard = A.Compose(
    [
        A.OneOf([A.RandomShadow(), A.RGBShift(r_shift_limit=40, g_shift_limit=40, b_shift_limit=40),
                 A.RandomBrightnessContrast(), A.HueSaturationValue(), A.ChannelShuffle()]),
        A.ShiftScaleRotate(shift_limit=0.03, rotate_limit=25),
        A.HorizontalFlip(p=0.1),
        A.ElasticTransform(p=0.1),
        A.ToGray(p=0.2),
        A.Blur(),
        A.GaussNoise(),
        A.Cutout(p=0.3, num_holes=16, max_h_size=16, max_w_size=16),
        RandomAugMix(severity=3, width=3, alpha=1., p=0.1),
    ]
)

transform_old = Compose([
    GaussNoise(var_limit=(10, 80), p=0.4),
    OneOf([
        MotionBlur(blur_limit=7, p=0.4),
        MedianBlur(blur_limit=7, p=0.4),
        Blur(blur_limit=7, p=0.4),
    ], p=0.2),
    ShiftScaleRotate(shift_limit=0.09, scale_limit=0.2, rotate_limit=8),
    OneOf([
        OpticalDistortion(p=0.3),
        GridDistortion(p=0.3),
        IAAPiecewiseAffine(p=0.3, nb_rows=2, nb_cols=2),
    ], p=0.2),
    OneOf([
        CLAHE(clip_limit=3),
        IAASharpen(),
        IAAEmboss(),
        RandomBrightnessContrast(brightness_limit=(-0.6, 0.7), contrast_limit=(-0.6, 0.6)),
    ], p=0.25),
    OneOf([
        ChannelDropout(channel_drop_range=(1, 1), fill_value=0, p=0.4),
        HueSaturationValue(p=0.4),
        InvertImg(p=0.2),
        ChannelShuffle(p=0.5)
    ], p=0.25),
    OneOf([
        MultiplicativeNoise(multiplier=[0.8, 1.2], per_channel=True, p=0.4),
        MultiplicativeNoise(multiplier=[0.8, 1.2], elementwise=True, p=0.4),
        MultiplicativeNoise(multiplier=[0.8, 1.2], elementwise=True, per_channel=True, p=0.4),
    ], p=0.25),
    JpegCompression(quality_lower=60, quality_upper=90, p=0.4)
], p=0.25)
