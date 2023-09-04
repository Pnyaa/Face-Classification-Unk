import logging
import os
from accelerate import Accelerator
import pandas as pd
import numpy as np
from PIL import Image
import torch
from transformers import ViTForImageClassification
from tqdm import tqdm
import lightning.pytorch as pl

from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField
from ffcv.loader import Loader, OrderOption
import ffcv.transforms as fftr
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder


batch_size = 1
grad_acc = 1
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


class TestDataset():
    def __init__(self, img_dir, size):
        self.img_dir = img_dir
        self.size = size

    def __len__(self):
        return 9999

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, f"{idx:06}.jpg")
        image = Image.open(img_path)
        image = image.resize((self.size, self.size), Image.Resampling.BICUBIC)

        return image, idx


def get_loaders(size, Imagenet=False):
    if Imagenet:
        data_mean, data_std = np.array([0.485, 0.456, 0.406]) * 255, np.array([0.229, 0.224, 0.225]) * 255
    else:
        data_mean, data_std = np.array([0.5550244450569153, 0.4250235855579376, 0.36004188656806946]) * 255, np.array([0.28600722551345825, 0.24972566962242126, 0.23863893747329712]) * 255
    testet = TestDataset("./test/", size)

    # Convert datasets to ffcv
    file_test = f"./ffcv/test_{size}_Imagenet.beton" if Imagenet else f"./ffcv/test_{size}.beton"
    if not os.path.isfile(file_test):
        writer_test = DatasetWriter(file_test,
                                    {'image': RGBImageField(write_mode='jpg', jpeg_quality=100),
                                     'label': IntField()}, num_workers=workers)
        writer_test.from_indexed_dataset(testet)

    decoder = SimpleRGBImageDecoder()
    normalize = fftr.NormalizeImage(mean=data_mean, std=data_std, type=np.float32)
    image_test_pipeline = [decoder, fftr.ToTensor(), fftr.ToTorchImage(), fftr.ToDevice(accelerator.device, non_blocking=False), normalize]
    label_pipeline = [IntDecoder(), fftr.ToTensor(), fftr.Convert(torch.float16), fftr.ToDevice(accelerator.device, non_blocking=False)]

    # Pipeline for each data field
    pipeline_test = {'image': image_test_pipeline, 'label': label_pipeline}
    testloader = Loader(file_test, batch_size=batch_size, num_workers=workers,
                        order=OrderOption.SEQUENTIAL, pipelines=pipeline_test, drop_last=False)

    return testloader


def predict(model, dataloader):
    dataloader = tqdm(dataloader, leave=False)
    y_pred = []
    with torch.no_grad():
        for batch in dataloader:
            X, _ = batch
            output = model(X).logits
            y_pred.extend(torch.argmax(output, dim=1).int().tolist())
    return y_pred


def main():
    logging.basicConfig(style='{', format='{asctime} : {message}', datefmt="%c", level=logging.INFO)
    testloader = get_loaders(224, Imagenet=True)
    model = ViTForImageClassification.from_pretrained('facebook/dino-vits8', num_labels=2,
                                                      cache_dir="dino-vits8/", ignore_mismatched_sizes=True)
    checkpoint = torch.load("results/Dino_sing/best_model.pth")
    # Rename keys of saved model (named changed by acceleator wrapper)
    # https://huggingface.co/docs/accelerate/package_reference/accelerator#saving-and-loading would have been more efficient
    state_dict = {}
    for k, v in checkpoint['state_dict'].items():
        state_dict[k[10:]] = v
    model.load_state_dict(state_dict)

    model, testloader = accelerator.prepare(model, testloader)
    model.eval()
    y_pred = predict(model, testloader)
    print(len(y_pred))
    print(y_pred[:10])
    results = pd.DataFrame(y_pred, columns=['score'])
    results['score'] = results['score'].map(lambda x: -1 if x == 0 else 1)
    results.to_csv("prediction_Dino_sing_best_model.csv", header=None, index=None)


if __name__ == "__main__":
    main()
