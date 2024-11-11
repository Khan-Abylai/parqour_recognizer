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

logging.basicConfig(
    format='[%(asctime)s:] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger('training')


def train(arguments, gpu):
    net, start_epoch = utils.get_model(
        arguments,
        gpu
    )
    optimizer = torch.optim.AdamW(net.parameters(), lr=arguments.lr, amsgrad=True)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, min_lr=1e-6, verbose=True)
    criterion = torch.nn.CTCLoss(reduction='sum', zero_infinity=True)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.set_device(gpu)
    converter = StrLabelConverter(config.alphabet)
    train_dataset = utils.get_dataset(arguments.data_dir, 'train')
    val_dataset = utils.get_dataset(arguments.data_dir, 'val')

    data_loader_train = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config.n_cpu,
        pin_memory=True
    )

    data_loader_val = torch.utils.data.DataLoader(dataset=val_dataset,
                                                  batch_size=args.batch_size,
                                                  shuffle=False,
                                                  num_workers=config.n_cpu,
                                                  pin_memory=True)

    for epoch in range(start_epoch, arguments.epochs):
        train_mean_loss = 0
        progress_bar = tqdm(data_loader_train)
        i = 0
        total_train_images = 0
        correct_train_predictions = 0
        word_error_rate = 0
        for idx, (images, labels, filepaths) in enumerate(progress_bar):
            encoded_train_labels, length = converter.encode(labels)
            train_images = images.cuda()
            encoded_train_labels = encoded_train_labels.cuda()
            length = length.cuda()
            batch_size = length.shape[0]
            with amp.autocast():
                predictions = net(train_images)
                predictions = predictions.permute(1, 0, 2).contiguous()
                prediction_size = torch.LongTensor([predictions.size(0)]).repeat(batch_size)
                train_current_loss = criterion(predictions, encoded_train_labels, prediction_size, length)

            predictions = predictions.argmax(2).detach().cpu()
            predicted_train_labels = np.array(converter.decode(predictions, prediction_size, raw=False))
            correct_train_predictions += (predicted_train_labels == labels).sum()
            word_error_rate += wer_metric.wer(predicted_train_labels, labels)

            train_mean_loss = train_mean_loss * \
                              (total_train_images / (
                                      total_train_images + batch_size)) + train_current_loss * batch_size / (
                                      total_train_images + batch_size)

            total_train_images += batch_size
            progress_bar.set_description(
                '[{}/{}] Current Loss: {:.4f} Loss: {:.4f} Acc: {:.4f} WER: {:.4f}'.format(epoch,
                                                                                           config.epochs,
                                                                                           train_current_loss,
                                                                                           train_mean_loss,
                                                                                           correct_train_predictions / total_train_images,
                                                                                           word_error_rate / total_train_images))

            net.zero_grad()
            clip_grad_norm_(net.parameters(), 1)
            train_current_loss.backward()
            optimizer.step()

            if i == len(data_loader_train) - 1:
                val_mean_loss = 0
                val_accuracy = 0
                val_cls_accuracy = 0
                total_val_images = 0
                correct_val_predictions = 0
                net.eval()

                with torch.no_grad():
                    for val_images, val_labels, val_filepaths in data_loader_val:
                        encoded_val_labels, length = converter.encode(val_labels)
                        predictions = net(val_images.cuda())
                        predictions = predictions.permute(1, 0, 2).contiguous()
                        batch_size = length.shape[0]

                        prediction_size = torch.LongTensor([predictions.size(0)]).repeat(batch_size)
                        current_val_loss = criterion(predictions, encoded_val_labels, prediction_size, length)
                        predictions = predictions.argmax(2).detach().cpu()
                        predicted_val_labels = np.array(converter.decode(predictions, prediction_size, raw=False))
                        correct_val_predictions += (predicted_val_labels == val_labels).sum()

                        val_mean_loss = val_mean_loss * \
                                        (total_val_images / (
                                                total_val_images + batch_size)) + current_val_loss * batch_size / (
                                                total_val_images + batch_size)

                        total_val_images += batch_size
                val_accuracy = correct_val_predictions / total_val_images
                train_accuracy = correct_train_predictions / total_train_images

                epoch_description = '[{}/{}] Train: {:.4f}, Accuracy: {:.4f}, Val: {:.4f}, Accuracy: {:.4f}, lr: {}'.format(
                    epoch, config.epochs,
                    train_mean_loss,
                    train_accuracy,
                    val_mean_loss,
                    val_accuracy,
                    optimizer.param_groups[0]['lr'])
                progress_bar.set_description(epoch_description)
                net.train()
                scheduler.step(val_mean_loss)

                state = {
                    'epoch': epoch,
                    'state_dict': net.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
                utils.save_model(arguments.checkpoint_dir, state, epoch)
            i += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--batch_size', type=int, default=config.batch_size
    )
    parser.add_argument(
        '--num_worker', type=int, default=config.n_cpu
    )
    parser.add_argument(
        '--lr', type=float, default=config.lr
    )
    parser.add_argument(
        '--optim', type=str, default='Adam'
    )
    parser.add_argument(
        '--load_weights', type=int, default=42
    )
    parser.add_argument(
        '--epochs', type=int, default=config.epochs
    )
    parser.add_argument('--batch_multiplier', type=int, default=1,
                        help='actual batch size = batch_size * batch_multiplier (use when cuda out of memory)')
    parser.add_argument(
        '--checkpoint_dir', type=str, default=config.checkpoint_dir
    )
    parser.add_argument(
        '--checkpoint', type=str, default=config.checkpoint
    )

    parser.add_argument(
        '--data_dir', type=str, default=config.data_dir
    )
    args = parser.parse_args()


    random.seed(42)
    np.random.seed(42)

    train(args, 0)
