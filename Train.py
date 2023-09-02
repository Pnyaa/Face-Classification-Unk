import logging
import os
import os.path
import pandas as pd
import numpy as np
from accelerate import Accelerator
from PIL import Image
from sklearn.model_selection import train_test_split
from sing import SING
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import lightning.pytorch as pl
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup, ViTForImageClassification

from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField
from ffcv.loader import Loader, OrderOption
import ffcv.transforms as fftr
from ffcv.fields.decoders import IntDecoder, RandomResizedCropRGBImageDecoder, SimpleRGBImageDecoder


batch_size = 64
grad_acc = 1
lr = 5e-4
weight_decay = 5e-3
writer = SummaryWriter(f"./logs/Dino_sing/lr{lr}_wd{weight_decay}_bs{batch_size*grad_acc}")
workers = 8
accelerator = Accelerator(gradient_accumulation_steps=grad_acc)
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')
pl.seed_everything(42, workers=True)


class IdemiaDataset():
    def __init__(self, data, img_dir, size):
        self.img_dir = img_dir
        self.data = data
        self.size = size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, label, genre, _ = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path)
        image = image.resize((self.size, self.size), Image.Resampling.BICUBIC)

        return image, label, genre


def get_loaders(labels, size, Imagenet=False):
    if Imagenet:
        data_mean, data_std = np.array([0.485, 0.456, 0.406]) * 255, np.array([0.229, 0.224, 0.225]) * 255
    else:
        data_mean, data_std = np.array([0.5550244450569153, 0.4250235855579376, 0.36004188656806946]) * 255, np.array([0.28600722551345825, 0.24972566962242126, 0.23863893747329712]) * 255
    data_train, data_val = train_test_split(labels, stratify=labels.stratif, test_size=0.2, random_state=35, shuffle=True)
    trainset = IdemiaDataset(data_train, "./train/", size)
    valset = IdemiaDataset(data_val, "./train/", size)

    # Convert datasets to ffcv
    file_train = f"./ffcv/train_{size}_Imagenet.beton" if Imagenet else f"./ffcv/train_{size}.beton"
    file_val = f"./ffcv/val_{size}_Imagenet.beton" if Imagenet else f"./ffcv/val_{size}.beton"
    if not os.path.isfile(file_train):
        writer_train = DatasetWriter(file_train, {
            'image': RGBImageField(write_mode='jpg', jpeg_quality=100),
            'label': IntField(),
            'genre': IntField()}, num_workers=workers)
        writer_train.from_indexed_dataset(trainset)

    if not os.path.isfile(file_val):
        writer_val = DatasetWriter(file_val, {
            'image': RGBImageField(write_mode='jpg', jpeg_quality=100),
            'label': IntField(),
            'genre': IntField()}, num_workers=workers)
        writer_val.from_indexed_dataset(valset)

    decoder = RandomResizedCropRGBImageDecoder((size, size), scale=(0.5, 1))
    decoder_val = SimpleRGBImageDecoder()
    normalize = fftr.NormalizeImage(mean=data_mean, std=data_std, type=np.float32)
    eraser_mean = list((np.random.rand(3,) * 255).astype(int))
    image_train_pipeline = [decoder, fftr.RandomHorizontalFlip(), fftr.Rotate(angle=0.5),
                            fftr.RandomErasing(erase_prob=0.5, scale=(0.02, 0.4), ratio=(0.15, 6), mean=eraser_mean),
                            fftr.RandomColorJitter(jitter_prob=0.5, brightness=[1.0, 1.5], contrast=0, saturation=[1.0, 1.5], hue=0),
                            fftr.GaussianBlur(0.5), fftr.ToTensor(), fftr.ToTorchImage(), fftr.ToDevice(accelerator.device, non_blocking=True), normalize]
    image_val_pipeline = [decoder_val, fftr.ToTensor(), fftr.ToTorchImage(), fftr.ToDevice(accelerator.device, non_blocking=True), normalize]
    label_pipeline = [IntDecoder(), fftr.ToTensor(), fftr.Convert(torch.float16), fftr.ToDevice(accelerator.device, non_blocking=True), fftr.Squeeze()]

    # Pipeline for each data field
    pipeline_train = {
        'image': image_train_pipeline, 'label': label_pipeline, 'genre': label_pipeline
    }
    pipeline_val = {
        'image': image_val_pipeline, 'label': label_pipeline, 'genre': label_pipeline
    }

    trainloader = Loader(file_train, batch_size=batch_size, num_workers=workers,
                         order=OrderOption.RANDOM, pipelines=pipeline_train)
    valloader = Loader(file_val, batch_size=batch_size, num_workers=workers,
                       order=OrderOption.SEQUENTIAL, pipelines=pipeline_val)

    return trainloader, valloader


def evaluate(model, valloader):
    valloader = tqdm(valloader, leave=False)
    with torch.no_grad():
        loss, count, global_acc = 0, 0, 0
        acc_g0, acc_g1, count_g0, count_g1 = 0, 0, 0, 0
        for batch in valloader:
            X_val, y_val, genre = batch
            count += X_val.shape[0]
            count_g0 += torch.sum(genre == -1)
            count_g1 += torch.sum(genre == 1)
            mask0 = (genre == -1).nonzero(as_tuple=True)[0]
            mask1 = (genre == 1).nonzero(as_tuple=True)[0]
            output = model(X_val).logits
            y_smoothed = torch.where(nn.functional.one_hot(y_val.long(), num_classes=2) == 1, 0.9, 0.1)
            loss += nn.functional.binary_cross_entropy_with_logits(output, y_smoothed, reduction="sum")
            y_pred = torch.argmax(output, dim=1)
            global_acc += torch.sum(y_pred == y_val)
            acc_g0 += torch.sum(y_pred[mask0] == y_val[mask0])
            acc_g1 += torch.sum(y_pred[mask1] == y_val[mask1])
            valloader.set_description(f"Val Loss: {loss/count:.4f} - Val acc: {global_acc/count:.4f} - Acc g0: {acc_g0/count_g0:.4f} - Acc g1: {acc_g1/count_g1:.4f}")
        acc_g0, acc_g1 = acc_g0 / count_g0, acc_g1 / count_g1
        score = 0.5 * (acc_g0 + acc_g1) - abs(acc_g0 - acc_g1)

    return loss / count, global_acc / count, acc_g0, acc_g1, score


def train(epoch, trainloader, valloader, model, optimizer, scheduler):
    epochs = tqdm(range(epoch))
    best_acc = 0
    for e in epochs:
        model.train()
        losses, count = 0, 0
        trainloader = tqdm(trainloader, leave=False)
        for idx, batch in enumerate(trainloader):
            with accelerator.accumulate(model):
                X, y, _ = batch
                count += X.shape[0]
                y_pred = model(X).logits
                y_smoothed = torch.where(nn.functional.one_hot(y.long(), num_classes=2) == 1, 0.9, 0.1)
                loss = nn.functional.binary_cross_entropy_with_logits(y_pred, y_smoothed, reduction="sum")
                losses += loss.item()
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                grads = [param.grad.detach().flatten() for param in model.parameters() if param.grad is not None]
                norm = torch.cat(grads).norm()
                optimizer.zero_grad()
                # Observe evolution of gradient norm at each step
                writer.add_scalar('Dino_sing/gradient_norm', norm, idx + e * len(trainloader) // grad_acc)
                trainloader.set_description(f"Train Iter: {idx+1}/{len(trainloader)} - LR: {scheduler.get_last_lr()[0]:e} - Gradient norm: {norm:e} - Total Loss: {losses/count:.4f}")

        model.eval()
        valloss, valacc, acc_g0, acc_g1, score = evaluate(model, valloader)
        epochs.set_description(f"Epoch: {e+1}/{epoch} - Train Loss: {losses/count:.4f} - Val loss: {valloss:.4f} - Val accuracy: {valacc:.4f} - Acc g0:{acc_g0:.4f} - Acc g1:{acc_g1:.4f} - Val score: {score:.4f}")
        writer.add_scalars('Dino_sing/learning_curves', {'train': losses / count, 'val': valloss}, e)
        writer.add_scalar('Dino_sing/learning_rate', scheduler.get_last_lr()[0], e)
        writer.add_scalars('Dino_sing/score', {'acc_g0': acc_g0, 'acc_g1': acc_g1, 'score': score}, e)
        if valacc > best_acc:
            best_acc = valacc
        torch.save({'epoch': e + 1, 'state_dict': model.state_dict(), 'score': score, 'accuracy': valacc,
                    'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()},
                    f"./results/Dino_sing/lr{lr}_wd{weight_decay}_bs{batch_size*grad_acc}_epoch{e}.pth")
    return best_acc


def main():
    logging.basicConfig(style='{', format='{asctime} : {message}', datefmt="%c", level=logging.WARNING)
    labels = pd.read_csv('./train.txt', sep='\t', header=None, names=['image', 'label', 'genre'])
    labels['label'] = labels['label'].map(lambda x: 0 if x == -1 else x)
    # Brutal hack : https://stackoverflow.com/a/51525992
    labels['stratif'] = labels['label'].astype(str) + labels['genre'].astype(str)
    trainloader, valloader = get_loaders(labels, 224, Imagenet=True)

    epoch = 10
    model = ViTForImageClassification.from_pretrained('facebook/dino-vits8', num_labels=2,
                                                      cache_dir="dino-vits8/", ignore_mismatched_sizes=True)
    no_decay_list = ["bias", "bn", "norm", "layernorm"]
    decay = [param for name, param in model.named_parameters() if not any(word in name for word in no_decay_list)]
    no_decay = [param for name, param in model.named_parameters() if any(word in name for word in no_decay_list)]
    parameters = [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    optim = SING(parameters, lr=lr)
    scheduler = get_cosine_schedule_with_warmup(optim, 0.05 * len(trainloader) * epoch / grad_acc, len(trainloader) * epoch / grad_acc)
    model, optim, scheduler = accelerator.prepare(model, optim, scheduler)
    # Juste in case : https://github.com/AdrienCourtois/SING/tree/main#further-recommandations
    # for module in model.modules():
    #     if isinstance(module, nn.LayerNorm):
    #         module.weight.requires_grad_(False)
    #         module.bias.requires_grad_(False)
    best_acc = train(epoch, trainloader, valloader, model, optim, scheduler)
    print(f"Best accuracy on the validation set: {best_acc}")


if __name__ == "__main__":
    main()
