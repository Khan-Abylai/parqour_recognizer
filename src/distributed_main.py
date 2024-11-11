import argparse
import os
import random
import logging
import torch
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch import optim
import torch.distributed as dist
from tqdm import tqdm
import torch.multiprocessing as mp
import torch.nn.functional as F

import argparse
import logging
import random
import warnings

import numpy as np
import torch
from torch.cuda import amp
from torch.nn.utils import clip_grad_norm_
from torch.optim import lr_scheduler
from tqdm import tqdm

from config import base_config as config
from utils import utils, wer_metric
from utils.converter import StrLabelConverter

warnings.filterwarnings("ignore")

logging.basicConfig(format='[%(asctime)s:] %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
logger = logging.getLogger('training')


def train(gpu, arguments):
    dist.init_process_group(backend='nccl', init_method='env://', world_size=arguments.gpu_nums, rank=gpu)
    torch.cuda.set_device(gpu)

    model_directory = arguments.checkpoint_dir + arguments.model_name + '/'
    model = utils.get_model(arguments, gpu=gpu)
    model, start_epochs = utils.load_weights(model, model_directory, arguments.checkpoint)
    optimizer = torch.optim.AdamW(model.parameters(), lr=arguments.lr, amsgrad=True)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, min_lr=1e-6, verbose=True)
    criterion = torch.nn.CTCLoss(reduction='sum', zero_infinity=True)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    converter = StrLabelConverter(config.alphabet)
    train_dataset = utils.get_dataset(arguments.data_dir, 'train')
    val_dataset = utils.get_dataset(arguments.data_dir, 'val')
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=arguments.gpu_nums,
                                                                    rank=gpu, shuffle=False)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=arguments.gpu_nums,
                                                                  rank=gpu, shuffle=False)
    train_dataloader = DataLoader(train_dataset, batch_size=arguments.batch_size, shuffle=False,
                                  num_workers=arguments.num_worker // arguments.gpu_nums, sampler=train_sampler,
                                  drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=arguments.batch_size, shuffle=False,
                                num_workers=arguments.num_worker // arguments.gpu_nums, sampler=val_sampler,
                                drop_last=True)
    for epoch in range(start_epochs, arguments.num_epochs):
        train_mean_loss = 0
        i = 0
        n = 0
        total_train_images = 0
        correct_train_predictions = 0
        word_error_rate = 0
        progress_bar = train_dataloader

        if gpu == 0:
            progress_bar = tqdm(train_dataloader)
        for idx, (images, labels, filepaths) in enumerate(progress_bar):
            encoded_train_labels, length = converter.encode(labels)
            train_images = images.cuda()
            encoded_train_labels = encoded_train_labels.cuda()
            length = length.cuda()
            batch_size = length.shape[0]
            with amp.autocast():
                predictions = model(train_images)
                predictions = predictions.permute(1, 0, 2).contiguous()
                prediction_size = torch.LongTensor([predictions.size(0)]).repeat(batch_size)
                train_current_loss = criterion(predictions, encoded_train_labels, prediction_size, length)
            predictions = predictions.argmax(2).detach().cpu()
            predicted_train_labels = np.array(converter.decode(predictions, prediction_size, raw=False))
            correct_train_predictions += (predicted_train_labels == labels).sum()
            word_error_rate += wer_metric.wer(predicted_train_labels, labels)
            train_mean_loss = train_mean_loss * (
                    total_train_images / (total_train_images + batch_size)) + train_current_loss * batch_size / (
                                      total_train_images + batch_size)

            total_train_images += batch_size
            epoch_desc = '[{}/{}] Current Loss: {:.4f} Loss: {:.4f} Acc: {:.4f} WER: {:.4f}'.format(epoch,
                                                                                                    config.epochs,
                                                                                                    train_current_loss,
                                                                                                    train_mean_loss,
                                                                                                    correct_train_predictions / total_train_images,
                                                                                                    word_error_rate / total_train_images)
            i += 1
            if gpu == 0:
                # logger.info(epoch_desc)
                progress_bar.set_description(epoch_desc)

            model.zero_grad()
            clip_grad_norm_(model.parameters(), 1)
            train_current_loss.backward()
            optimizer.step()

            if i == len(train_dataloader) - 1:
                val_mean_loss = 0
                val_accuracy = 0
                val_cls_accuracy = 0
                total_val_images = 0
                correct_val_predictions = 0
                model.eval()

                with torch.no_grad():
                    for val_images, val_labels, val_filepaths in val_dataloader:
                        encoded_val_labels, length = converter.encode(val_labels)
                        predictions = model(val_images.cuda())
                        predictions = predictions.permute(1, 0, 2).contiguous()
                        batch_size = length.shape[0]

                        prediction_size = torch.LongTensor([predictions.size(0)]).repeat(batch_size)
                        current_val_loss = criterion(predictions, encoded_val_labels, prediction_size, length)
                        predictions = predictions.argmax(2).detach().cpu()
                        predicted_val_labels = np.array(converter.decode(predictions, prediction_size, raw=False))
                        correct_val_predictions += (predicted_val_labels == val_labels).sum()
                        val_mean_loss = val_mean_loss * (total_val_images / (
                                total_val_images + batch_size)) + current_val_loss * batch_size / (
                                                total_val_images + batch_size)

                        total_val_images += batch_size
                val_accuracy = correct_val_predictions / total_val_images
                train_accuracy = correct_train_predictions / total_train_images
                epoch_description = '{}/{} Train: {:.4f}, Accuracy: {:.4f}, Val: {:.4f}, Accuracy: {:.4f}, lr: {}'.format(
                    epoch, config.epochs, train_mean_loss, train_accuracy, val_mean_loss, val_accuracy,
                    optimizer.param_groups[0]['lr'])

                if gpu == 0:
                    progress_bar.set_description(epoch_description)
                    state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
                    torch.save(state, model_directory + epoch_description.replace(' ', '_').replace('/',
                                                                                                    '_') + '.pth')  # logger.info(epoch_description)

                model.train()
                scheduler.step(val_mean_loss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_h', type=int, default=config.img_h)
    parser.add_argument('--num_class', type=int, default=config.num_class)
    parser.add_argument('--model_lstm_layers', type=int, default=config.model_lstm_layers)
    parser.add_argument('--model_lsrm_is_bidirectional', type=int, default=config.model_lsrm_is_bidirectional)
    parser.add_argument('--batch_size', type=int, default=config.batch_size)
    parser.add_argument('--num_worker', type=int, default=config.n_cpu)
    parser.add_argument('--lr', type=float, default=config.lr)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--num_epochs', type=int, default=config.epochs)
    parser.add_argument('--gpu_nums', type=int, default=torch.cuda.device_count())
    parser.add_argument('--checkpoint_dir', type=str, default=config.checkpoint_dir)
    parser.add_argument('--data_dir', type=str, default=config.data_dir)
    parser.add_argument('--model_name', type=str, default=config.model_name)

    args = parser.parse_args()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    random.seed(42)
    np.random.seed(42)

    mp.spawn(train, nprocs=args.gpu_nums, args=(args,))
