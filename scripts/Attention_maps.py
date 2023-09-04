import logging
import os
from accelerate import Accelerator
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torchvision
from transformers import ViTForImageClassification
from tqdm import tqdm
import lightning.pytorch as pl

from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField
from ffcv.loader import Loader, OrderOption
import ffcv.transforms as fftr
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder


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
        img_name, _, _ = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path)
        image = image.resize((self.size, self.size), Image.Resampling.BICUBIC)

        return image, idx


def get_loaders(data, size, Imagenet=False):
    if Imagenet:
        data_mean, data_std = np.array([0.485, 0.456, 0.406]) * 255, np.array([0.229, 0.224, 0.225]) * 255
    else:
        data_mean, data_std = np.array([0.5550244450569153, 0.4250235855579376, 0.36004188656806946]) * 255, np.array([0.28600722551345825, 0.24972566962242126, 0.23863893747329712]) * 255
    dataset = IdemiaDataset(data, "./train/", size)

    # Convert datasets to ffcv
    file = f"./ffcv/sample_{size}_Imagenet.beton" if Imagenet else f"./ffcv/sample_{size}.beton"
    writer_test = DatasetWriter(file,
                                {'image': RGBImageField(write_mode='jpg', jpeg_quality=100),
                                 'label': IntField()}, num_workers=workers)
    writer_test.from_indexed_dataset(dataset)

    decoder = SimpleRGBImageDecoder()
    normalize = fftr.NormalizeImage(mean=data_mean, std=data_std, type=np.float32)
    image_test_pipeline = [decoder, fftr.ToTensor(), fftr.ToTorchImage(), fftr.ToDevice(accelerator.device, non_blocking=False), normalize]
    label_pipeline = [IntDecoder(), fftr.ToTensor(), fftr.Convert(torch.float16), fftr.ToDevice(accelerator.device, non_blocking=False)]

    # Pipeline for each data field
    pipeline_test = {'image': image_test_pipeline, 'label': label_pipeline}
    dataloader = Loader(file, batch_size=len(data), num_workers=workers,
                        order=OrderOption.SEQUENTIAL, pipelines=pipeline_test, drop_last=False)

    return dataloader


def main():
    logging.basicConfig(style='{', format='{asctime} : {message}', datefmt="%c", level=logging.INFO)
    labels = pd.read_csv('./train.txt', sep='\t', header=None, names=['image', 'label', 'genre'])
    labels['label'] = labels['label'].map(lambda x: 0 if x == -1 else x)
    samples = np.load("attention_maps/idx_samples.npy")
    data = labels.iloc[samples]
    dataloader = get_loaders(data, 224, Imagenet=True)
    model = ViTForImageClassification.from_pretrained('facebook/dino-vits8', num_labels=2,
                                                      cache_dir="dino-vits8/", ignore_mismatched_sizes=True)
    checkpoint = torch.load("results/Dino_sing/best_model.pth")
    # Rename keys of saved model (named changed by acceleator wrapper)
    # https://huggingface.co/docs/accelerate/package_reference/accelerator#saving-and-loading would have been more efficient
    state_dict = {}
    for k, v in checkpoint['state_dict'].items():
        state_dict[k[10:]] = v
    model.load_state_dict(state_dict)
    model, dataloader = accelerator.prepare(model, dataloader)
    model.eval()

    for batch in dataloader:
        X, _ = batch
    outputs = model(X, output_attentions=True)
    last_attention = outputs.attentions[-1]
    # https://github.com/facebookresearch/dino/blob/main/visualize_attention.py -> récupère les nh attention maps pour UNE SEULE image
    # Dimension de l'image divisée par taille de patch (vits8 dans notre cas, donc patch_size = 8)
    featmap_dim = 224 // 8
    # Number of heads
    nh = last_attention.shape[1]
    for i, attention in enumerate(tqdm(last_attention)):
        # We only keep the output patch attention
        attention = attention[:, 0, 1:].reshape(nh, featmap_dim, featmap_dim)
        attention = torch.nn.functional.interpolate(attention.unsqueeze(0), scale_factor=8, mode="bicubic")[0].cpu().detach().numpy()
        torchvision.utils.save_image(torchvision.utils.make_grid(X[i].cpu().detach(), normalize=True, scale_each=True),
                                     os.path.join("attention_maps", f"img{i}.png"))
        attention_mean = []
        for j in range(nh):
            fname = os.path.join("attention_maps", f"img{i}_head{j}.png")
            attention_mean.append(attention[j])
            plt.imsave(fname=fname, arr=attention[j], format='png', cmap="inferno")
        attention_mean = np.stack(attention_mean)
        fname = os.path.join("attention_maps", f"img{i}_heatmap.png")
        plt.imsave(fname=fname, arr=np.mean(attention_mean, axis=0), format='png', cmap="inferno")


if __name__ == "__main__":
    main()
