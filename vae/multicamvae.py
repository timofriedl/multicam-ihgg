import abc
import sys
from math import ceil

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from vae.coordbdvae import CoordBDVAE
from vae.innervae import InnerVae
from vae.trainer import Trainer
from vae.utils import image_to_tensor, tensor_to_image, images_to_tensor, tensor_wrap, \
    images_to_tensor_cpu, device


class MultiCamVae:
    def __init__(self, num_cams, width, height, latent_dim: int):
        self.num_cams = num_cams
        self.width = width
        self.height = height
        self.latent_dim = latent_dim

    @abc.abstractmethod
    def encode(self, images: np.ndarray) -> Tensor:
        return

    @abc.abstractmethod
    def decode(self, latent: Tensor) -> np.ndarray:
        return

    def forward(self, images: np.ndarray) -> np.ndarray:
        return self.decode(self.encode(images))

    @abc.abstractmethod
    def train(self, dataset: np.ndarray, model_name, batch_size, epochs, lefe, lefe_dataset):
        return

    @staticmethod
    @abc.abstractmethod
    def load(base_path, num_cams, width, height, latent_dim: int):
        return


class ConcatEncodeVae(MultiCamVae):
    def __init__(self, num_cams, width, height, latent_dim: int):
        super().__init__(num_cams, width, height, latent_dim)
        self.vae = CoordBDVAE(width * num_cams, height, latent_dim)
        self.inner_lat_dim = latent_dim

    def encode(self, images: np.ndarray) -> Tensor:
        single_image = np.concatenate(images, axis=1)
        mu, logvar = self.vae.encode(image_to_tensor(single_image))
        return self.vae.sample(mu, logvar)[0]

    def decode(self, latent: Tensor) -> np.ndarray:
        concat_image = tensor_to_image(self.vae.decode(tensor_wrap(latent)))
        return np.array(np.hsplit(concat_image, self.num_cams))

    def train(self, dataset: np.ndarray, model_name, batch_size, epochs, lefe=False, lefe_dataset=None):
        print("Concatenating images...")
        concat_dataset = np.empty([dataset.shape[0], self.vae.height, self.vae.width, 3])
        if lefe:
            concat_lefe_dataset = np.empty(concat_dataset.shape)

        for i in tqdm(range(dataset.shape[0])):
            concat_dataset[i] = np.concatenate(dataset[i, :self.num_cams], axis=1)
            if lefe:
                concat_lefe_dataset[i] = np.concatenate(lefe_dataset[i, :self.num_cams], axis=1)

        print("Converting to tensor...")
        data = images_to_tensor(concat_dataset)
        dl = DataLoader(data, batch_size=batch_size, shuffle=not lefe)
        if lefe:
            lefe_data = images_to_tensor(concat_lefe_dataset)
            lefe_dl = DataLoader(lefe_data, batch_size=batch_size, shuffle=False)

        print("Training...")
        beta = 10. if "pick_and_place" in model_name else 1.
        Trainer.train_vae(dl, self.vae, model_name, epochs=epochs, bar_log=True, goal_dataset=lefe_dl if lefe else dl,
                          beta=beta)

    @staticmethod
    def load(base_path, num_cams):
        coordbdvae = torch.load(base_path + ".pt")
        mvae = ConcatEncodeVae(num_cams, coordbdvae.width // num_cams, coordbdvae.height, coordbdvae.latent_dim)
        mvae.vae = coordbdvae
        return mvae


class EncodeConcatVae(MultiCamVae):
    def __init__(self, num_cams, width, height, latent_dim: int):
        super().__init__(num_cams, width, height, latent_dim)
        self.vaes = list(map(lambda _: CoordBDVAE(width, height, latent_dim), range(num_cams)))
        self.inner_lat_dim = latent_dim * self.num_cams

    def encode(self, images: np.ndarray) -> Tensor:
        images = images_to_tensor(images)

        latent = torch.empty(self.inner_lat_dim)
        for c in range(self.num_cams):
            mu, logvar = self.vaes[c].encode(tensor_wrap(images[c]))
            z = self.vaes[c].sample(mu, logvar)[0]
            latent[(c * self.latent_dim):((c + 1) * self.latent_dim)] = z

        return latent

    def decode(self, latent: Tensor) -> np.ndarray:
        latent = latent.reshape(self.num_cams, self.latent_dim).to(device)

        images = np.empty([self.num_cams, self.height, self.width, 3], dtype=np.uint8)
        for c in range(self.num_cams):
            images[c] = tensor_to_image(self.vaes[c].decode(tensor_wrap(latent[c])))

        return images

    def train(self, dataset: np.ndarray, model_name, batch_size, epochs, lefe=False, lefe_dataset=None):
        for c in range(self.num_cams):
            print("VAE %d/%d:" % (c + 1, self.num_cams))

            print("Initializing dataset...")
            data = images_to_tensor(dataset[:, c])
            dl = DataLoader(data, batch_size=batch_size, shuffle=not lefe)

            if lefe:
                lefe_data = images_to_tensor(lefe_dataset[:, c])
                lefe_dl = DataLoader(lefe_data, batch_size=batch_size, shuffle=False)

            print("Training...")
            beta = 10. if "pick_and_place" in model_name else 1.
            Trainer.train_vae(dl, self.vaes[c], model_name="{}_{}".format(model_name, c),
                              goal_dataset=lefe_dl if lefe else dl,
                              epochs=epochs // self.num_cams, bar_log=True, beta=beta)

    @staticmethod
    def load(base_path, num_cams):
        vaes = list()
        for i in range(num_cams):
            vaes.append(torch.load("{}_{}.pt".format(base_path, i)))

        mvae = EncodeConcatVae(num_cams, vaes[0].width, vaes[0].height, vaes[0].latent_dim)
        mvae.vaes = vaes
        return mvae


class EncodeConcatEncodeVae(MultiCamVae):
    def __init__(self, num_cams, width, height, latent_dim: int):
        super().__init__(num_cams, width, height, latent_dim)
        self.vaes = list(map(lambda _: CoordBDVAE(width, height, latent_dim), range(num_cams)))
        self.inner_vae = InnerVae(input_dim=num_cams * latent_dim, latent_dim=latent_dim)
        self.inner_lat_dim = latent_dim

    def outer_encode(self, images: np.ndarray) -> Tensor:
        images = images_to_tensor_cpu(images)

        outer_latent = torch.empty(self.num_cams * self.latent_dim)
        for c in range(self.num_cams):
            model = self.vaes[c].to(device)
            mu, logvar = model.encode(tensor_wrap(images[c].to(device)))
            z = model.sample(mu, logvar)[0]

            outer_latent[(c * self.latent_dim):((c + 1) * self.latent_dim)] = z

        return outer_latent

    def outer_encode_all(self, images: np.ndarray) -> Tensor:
        images = images_to_tensor_cpu(images.reshape(images.shape[0] * self.num_cams, self.height, self.width, 3)) \
            .reshape(images.shape[0], self.num_cams, 3, self.height, self.width)

        outer_latent = torch.empty(images.shape[0], self.num_cams * self.latent_dim)
        for c in range(self.num_cams):
            model = self.vaes[c].to(device)
            mu, logvar = model.encode(images[:, c].to(device))
            z = model.sample(mu, logvar)

            min_pos = c * self.latent_dim
            max_pos = (c + 1) * self.latent_dim
            outer_latent[:, min_pos:max_pos] = z

        return outer_latent

    def inner_encode(self, outer_latent: Tensor) -> Tensor:
        model = self.inner_vae.to(device)
        mu, logvar = model.encode(tensor_wrap(outer_latent.to(device)))
        return model.sample(mu, logvar)[0]

    def encode(self, images: np.ndarray) -> Tensor:
        return self.inner_encode(self.outer_encode(images))

    def inner_decode(self, inner_latent: Tensor) -> Tensor:
        return self.inner_vae.decode(tensor_wrap(inner_latent))[0]

    def outer_decode(self, outer_latent: Tensor) -> np.ndarray:
        latent = outer_latent.reshape(self.num_cams, self.latent_dim)

        images = np.empty([self.num_cams, self.height, self.width, 3], dtype=np.uint8)
        for c in range(self.num_cams):
            images[c] = tensor_to_image(self.vaes[c].decode(tensor_wrap(latent[c])))

        return images

    def decode(self, latent: Tensor) -> np.ndarray:
        return self.outer_decode(self.inner_decode(latent))

    def train(self, dataset: np.ndarray, model_name, batch_size, epochs, lefe, lefe_dataset, skip_outer=False):
        for c in range(self.num_cams):
            print("VAE %d/%d:" % (c + 1, self.num_cams + 1))

            print("Initializing dataset...")
            data = images_to_tensor(dataset[:, c])
            dl = DataLoader(data, batch_size=batch_size, shuffle=not lefe)

            if lefe:
                lefe_data = images_to_tensor(lefe_dataset[:, c])
                lefe_dl = DataLoader(lefe_data, batch_size=batch_size, shuffle=False)

            print("Training...")
            ep = 0 if skip_outer else epochs // (self.num_cams + 1)
            beta = 10. if "pick_and_place" in model_name else 1.
            Trainer.train_vae(dl, self.vaes[c], model_name="{}_{}".format(model_name, c), epochs=ep, bar_log=True,
                              goal_dataset=lefe_dl if lefe else dl, beta=beta)

        print("Inner VAE:")
        print("Initializing dataset...")
        data = torch.empty(dataset.shape[0], self.num_cams * self.latent_dim, dtype=torch.float32)
        for b in tqdm(range(ceil(data.shape[0] / batch_size))):
            pos_min = b * batch_size
            pos_max = min(dataset.shape[0], (b + 1) * batch_size)
            data[pos_min:pos_max] = self.outer_encode_all(dataset[pos_min:pos_max])
        dl = DataLoader(data.to(device), batch_size=batch_size, shuffle=True)

        print("Training...")
        ep = epochs // (self.num_cams + 1)
        Trainer.train_vae(dl, self.inner_vae, model_name="{}_inner".format(model_name), epochs=ep, bar_log=True,
                          beta=5., save_images_every=sys.maxsize)

    @staticmethod
    def load(base_path, num_cams):
        vaes = list()
        for i in range(num_cams):
            vaes.append(torch.load("{}_{}.pt".format(base_path, i)))

        inner_vae = torch.load("{}_inner.pt".format(base_path))

        mvae = EncodeConcatEncodeVae(num_cams, vaes[0].width, vaes[0].height, vaes[0].latent_dim)
        mvae.vaes = vaes
        mvae.inner_vae = inner_vae
        return mvae
