import random

try:
    import src.config.base_config as config
except Exception:
    import config.base_config as config
import numpy as np
import torch
from torch.utils.data.dataset import Dataset


class CutMix(Dataset):
    def __init__(self, dataset, num_mix=1, beta=1., prob=1.0):
        self.dataset = dataset
        self.num_class = len(config.car_types)
        self.num_mix = num_mix
        self.beta = beta
        self.prob = prob
        self.car_types = dataset.car_types

    def __getitem__(self, index):
        img, y = self.dataset[index]
        y_onehot = onehot(self.num_class, y)

        for _ in range(self.num_mix):
            r = np.random.rand(1)
            if self.beta <= 0 or r > self.prob:
                continue

            # generate mixed sample
            lam = np.random.beta(self.beta, self.beta)

            rand_index = random.choice(range(len(self)))

            img2, y2 = self.dataset[rand_index]
            y2_onehot = onehot(self.num_class, y2)

            bbx1, bby1, bbx2, bby2 = rand_bbox(img.size(), lam)
            img[:, bbx1:bbx2, bby1:bby2] = img2[:, bbx1:bbx2, bby1:bby2]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img.size()[-1] * img.size()[-2]))

            y_onehot = y_onehot * lam + y2_onehot * (1. - lam)

        return img, torch.Tensor(y_onehot)

    def __len__(self):
        return len(self.dataset)


def cross_entropy_soft_targets(input, target, size_average=True):
    """ Cross entropy that accepts soft targets
    Args:
         pred: predictions for neural network
         targets: targets, can be soft
         size_average: if false, sum is returned instead of mean
    Examples::
        input = torch.FloatTensor([[1.1, 2.8, 1.3], [1.1, 2.1, 4.8]])
        input = torch.autograd.Variable(out, requires_grad=True)
        target = torch.FloatTensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
        target = torch.autograd.Variable(y1)
        loss = cross_entropy(input, target)
        loss.backward()
    """
    logsoftmax = torch.nn.LogSoftmax(dim=1)
    if size_average:
        return torch.mean(torch.sum(-target * logsoftmax(input), dim=1))
    else:
        return torch.sum(torch.sum(-target * logsoftmax(input), dim=1))


def onehot(size, target):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.
    return vec


def rand_bbox(size, lam):
    if len(size) == 4:
        w = size[2]
        h = size[3]
    elif len(size) == 3:
        w = size[1]
        h = size[2]
    else:
        raise Exception

    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(w * cut_rat)
    cut_h = np.int(h * cut_rat)

    # uniform
    cx = np.random.randint(w)
    cy = np.random.randint(h)

    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)

    return bbx1, bby1, bbx2, bby2
