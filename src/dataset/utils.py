import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
import os

try:
    import src.config.base_config as config
    from src.scripts.random_string_generator import get_random_string_generator
except:
    import config.base_config as config
    from scripts.random_string_generator import get_random_string_generator


def padding(img, expected_size):
    desired_size = expected_size
    delta_width = desired_size - img.size[0]
    delta_height = desired_size - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)


def resize_with_padding(img, expected_size):
    img.thumbnail((expected_size[0], expected_size[1]))
    # print(img.size)
    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)


def preprocess(img, transform=None, imwrite=None):
    original_image = img.copy()

    x = cv2.resize(original_image, (config.img_w, config.img_h))
    if transform is not None:
        x = transform(image=x)['image']

    h, w, _ = img.shape
    left = 0
    right = 0
    top = 0
    bottom = 0
    if w > h:
        h = int(h * config.img_w / w)
        w = config.img_w
        top = (config.img_h - h) // 2
        bottom = config.img_h - h - top
    else:
        w = int(w * config.img_h / h)
        h = config.img_h
        left = (config.img_w - w) // 2
        right = config.img_w - w - left
    if top < 0 or bottom < 0 or left < 0 or right < 0:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (w, h))
        im_pil = Image.fromarray(img)
        img = resize_with_padding(im_pil, (config.img_w, config.img_h))
        x = np.asarray(img)
        if transform is not None:
            x = transform(image=x)['image']
    else:
        x = cv2.resize(img, (w, h))
        if transform is not None:
            x = transform(image=x)['image']
        x = cv2.copyMakeBorder(x, top, bottom, left, right,
                               cv2.BORDER_CONSTANT, value=(0, 0, 0))
    # if imwrite is not None and imwrite:
    #     f_name = get_random_string_generator(10, postfix="-1.jpg")
    #     second_f_name = f_name.replace("-1.jpg", "-2.jpg")
    #     path = os.path.join("/home/user/parking_recognizer/debug/test_images", f_name)
    #     path_2 = os.path.join("/home/user/parking_recognizer/debug/test_images", second_f_name)
    #     cv2.imwrite(path, x)
    #     cv2.imwrite(path_2, original_image)

    x = x.astype(np.float32) / 255.
    x = x.transpose(2, 0, 1)
    x = torch.tensor(x)

    return x
