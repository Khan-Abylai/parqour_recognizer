import argparse
import logging
import random
import warnings
import albumentations as A
import numpy as np
import torch
from torch.cuda import amp
from torch.nn.utils import clip_grad_norm_
from torch.optim import lr_scheduler
from tqdm import tqdm
from models.base_model import CRNN_2
from config import base_config as config
from utils import utils, wer_metric
from utils.converter import StrLabelConverter, RegionConverter
from augmentation.transforms import transform_old, transform_album_hard
from dataset.lp_dataset import LPRegionDataset

warnings.filterwarnings("ignore")

logging.basicConfig(format='[%(asctime)s:] %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
logger = logging.getLogger('training')


def train(arguments, gpu):
    base_model, start_epoch = utils.get_model(arguments.load_weights, arguments.checkpoint_dir, gpu)

    teacher_state_dict = torch.load(arguments.teacher_model_path)['state_dict']
    model = CRNN_2(image_h=config.img_h, num_class=config.num_class, num_layers=config.model_lstm_layers,
                   is_lstm_bidirectional=config.model_lsrm_is_bidirectional, num_regions=config.num_regions)
    model_dict = model.state_dict()

    keys = []
    for k, v in teacher_state_dict.items():
        if "classificator" in k:
            keys.append(k)

    for item in keys:
        del teacher_state_dict[item]
    pretrained_dict = {k: v for k, v in teacher_state_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    model.cuda()


    optimizer = torch.optim.AdamW(model.parameters(), lr=arguments.lr, amsgrad=True)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, min_lr=1e-6, verbose=True)
    criterion = torch.nn.CTCLoss(reduction='sum', zero_infinity=True)
    ce_loss = torch.nn.CrossEntropyLoss()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.set_device(gpu)
    converter = StrLabelConverter(config.alphabet)
    region_converter = RegionConverter(config.regions)

    train_param = {"train": True, "data_dir": arguments.data_dir, "csv_files": ["/home/user/data/experiment/fine_tuning_for_UAE/uae_data/train.csv"],
                   "transform": transform_old}
    val_param = {"train": False, "data_dir": arguments.data_dir, "csv_files": ["/home/user/data/experiment/fine_tuning_for_UAE/uae_data/val.csv"],
                 "transform": A.Compose([A.NoOp()])}

    train_dataset = LPRegionDataset(**train_param)
    val_dataset = LPRegionDataset(**val_param)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=config.n_cpu, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=config.n_cpu, pin_memory=True)

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, min_lr=1e-6, verbose=True)
    i = 0
    for epoch in range(start_epoch, config.epochs):
        train_mean_loss = 0
        progress_bar = tqdm(train_loader)

        total_train_images = 0
        correct_train_predictions = 0
        correct_cls_predicitons = 0
        word_error_rate = 0

        for idx, (images, labels, region, filepaths) in enumerate(progress_bar):
            # print(labels)
            region_label = region_converter.encode(region).cuda()
            encoded_train_labels, length = converter.encode(labels)
            train_images = images.cuda()
            encoded_train_labels = encoded_train_labels.cuda()
            length = length.cuda()
            batch_size = length.shape[0]
            with torch.enable_grad():
                predictions, classifier = model(train_images)
                predictions = predictions.permute(1, 0, 2).contiguous()
                prediction_size = torch.LongTensor([predictions.size(0)]).repeat(batch_size)
                train_current_loss = criterion(predictions, encoded_train_labels, prediction_size, length)
                train_current_loss_cls = ce_loss(classifier, region_label)

            predictions = predictions.argmax(2).detach().cpu()
            predicted_train_labels = np.array(converter.decode(predictions, prediction_size, raw=False))
            correct_train_predictions += (predicted_train_labels == labels).sum()
            correct_cls_predicitons += (classifier.argmax(1) == region_label).sum()
            word_error_rate += wer_metric.wer(predicted_train_labels, labels)

            train_current_loss += train_current_loss_cls
            train_mean_loss = train_mean_loss * (
                        total_train_images / (total_train_images + batch_size)) + train_current_loss * batch_size / (
                                      total_train_images + batch_size)

            total_train_images += batch_size

            progress_bar.set_description(
                '[{}/{}] Current Loss: {:.4f} Loss: {:.4f} Acc: {:.4f} Cls Acc: {:.4f} WER: {:.4f}'.format(epoch,
                                                                                                           config.epochs,
                                                                                                           train_current_loss,
                                                                                                           train_mean_loss,
                                                                                                           correct_train_predictions / total_train_images,
                                                                                                           correct_cls_predicitons / total_train_images,
                                                                                                           word_error_rate / total_train_images))

            model.zero_grad()
            clip_grad_norm_(model.parameters(), 1)

            train_current_loss.backward()
            optimizer.step()

            if i == len(train_loader) - 1:

                val_mean_loss = 0
                val_accuracy = 0
                val_cls_accuracy = 0
                total_val_images = 0
                correct_val_predictions = 0
                correct_val_classifier = 0
                model.eval()

                with torch.no_grad():
                    for val_images, val_labels, var_regions, _ in val_loader:
                        country_label_val = region_converter.encode(var_regions).cuda()
                        encoded_val_labels, length = converter.encode(val_labels)
                        predictions, classifier = model(val_images.cuda())
                        predictions = predictions.permute(1, 0, 2).contiguous()
                        batch_size = length.shape[0]
                        prediction_size = torch.LongTensor([predictions.size(0)]).repeat(batch_size)
                        current_val_loss = criterion(predictions, encoded_val_labels, prediction_size, length)
                        current_val_loss += ce_loss(classifier, country_label_val)

                        predictions = predictions.argmax(2).detach().cpu()
                        predicted_val_labels = np.array(converter.decode(predictions, prediction_size, raw=False))
                        correct_val_predictions += (predicted_val_labels == val_labels).sum()
                        correct_val_classifier += (country_label_val == classifier.argmax(1)).sum()

                        val_mean_loss = val_mean_loss * (total_val_images / (
                                total_val_images + batch_size)) + current_val_loss * batch_size / (
                                                total_val_images + batch_size)

                        total_val_images += batch_size

                val_accuracy = correct_val_predictions / total_val_images
                train_accuracy = correct_train_predictions / total_train_images
                val_cls_accuracy = correct_val_classifier / total_val_images

                epoch_description = '[{}/{}] Train: {:.4f}, Accuracy: {:.4f}, Val: {:.4f}, Accuracy: {:.4f}, Cls Acc: {:.4f}, lr: {}'.format(
                    epoch, config.epochs, train_mean_loss, train_accuracy, val_mean_loss, val_accuracy,
                    val_cls_accuracy, optimizer.param_groups[0]['lr'])

                progress_bar.set_description(epoch_description)
                model.train()
                scheduler.step(val_mean_loss)

            i += 1
        state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        print("saving model")
        utils.save_model("/home/user/data/experiment/fine_tuning_for_UAE/child_model", state, epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=196)
    parser.add_argument('--num_worker', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--optim', type=str, default='Adam')
    parser.add_argument('--load_weights', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_multiplier', type=int, default=1,
                        help='actual batch size = batch_size * batch_multiplier (use when cuda out of memory)')
    parser.add_argument('--checkpoint_dir', type=str,
                        default="/home/user/data/experiment/fine_tuning_for_UAE/model")
    parser.add_argument('--checkpoint', type=str, default="")
    parser.add_argument('--data_dir', type=str, default="/home/user/data/experiment/fine_tuning_for_UAE/uae_data")
    parser.add_argument("--teacher_model_path", type=str,
                        default='/home/user/data/experiment/fine_tuning_for_UAE/base_model_weights/epoch_53.pth')
    args = parser.parse_args()
    random.seed(42)
    np.random.seed(42)

    train(args, 0)
