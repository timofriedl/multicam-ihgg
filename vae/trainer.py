import shutil
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import Tensor, nn
from tqdm import tqdm

sys.path.append("..")

from vae.innervae import InnerVae
from vae.utils import save_examples, loss_fn_weighted, loss_fn_weighted2, device

images_path = "images"


class Trainer:
    @staticmethod
    def create_images_folder():
        path = Path(images_path)
        if path.exists():
            shutil.rmtree(images_path)
        path.mkdir()

    @staticmethod
    def save_img_examples(images: Tensor, name: str, epoch: int, it: int):
        save_examples(images, f'{name}', img_format="png")
        save_examples(images, f'{name}_{epoch}_{it}')

    @staticmethod
    def save_entanglement_img(entanglement_data: list, model: nn.Module, epoch: int, it: int):
        fig = plt.figure()

        for images in entanglement_data:
            mu, logvar = model.encode(images.to(device))
            latent = model.sample(mu, logvar).detach().cpu()
            plt.plot(latent[:, 0], latent[:, 1], 'o', markersize=3)

        plt.xlabel('latent x')
        plt.ylabel('latent y')
        epoch_str = format(epoch, '05d')
        it_str = format(it, '05d')
        fig.savefig(f'{images_path}/entanglement_{epoch_str}_{it_str}.jpeg')
        fig.savefig(f'{images_path}/entanglement.png')
        plt.close(fig)

    @staticmethod
    def train_vae(dataset, model, model_name, goal_dataset=None, model_folder="./models/", alpha=10., beta=1.,
                  bar_log=False,
                  log_every=10,
                  load=True, save_images_every=1000, save_every=10, start_epoch=0, epochs=500, entanglement_data=None):
        if not bar_log:
            print(f'Device: {device}')

        if load:
            try:
                model = torch.load(f"{model_folder}{model_name}.pt")
            except FileNotFoundError:
                Trainer.create_images_folder()
                print("Found no model to load. Created new.")

        if goal_dataset == None:
            goal_dataset = dataset

        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        epoch_range = range(start_epoch + 1, epochs + 1)
        for epoch in tqdm(epoch_range) if bar_log else epoch_range:
            if not bar_log:
                print(f'Epoch [{epoch}/{epochs}]')
            start_time = time.time()

            train_loss = 0
            train_mse = 0
            train_kl = 0

            for i, (d, goal_batch) in enumerate(zip(dataset, goal_dataset)):
                optimizer.zero_grad()

                out, mu, logvar = model(d)
                if model_name == 'bdvae_partial_reconstruction' or model_name == 'bdvae_full_reconstruction':
                    mse_loss, kl, loss = loss_fn_weighted(goal_batch, out, mu, logvar, alpha=alpha, beta=beta)
                else:
                    mse_loss, kl, loss = loss_fn_weighted2(goal_batch, out, mu, logvar, alpha=alpha, beta=beta)

                loss.backward(retain_graph=isinstance(model, InnerVae))
                optimizer.step()

                train_loss += loss.item()
                train_mse += mse_loss.item()
                train_kl += kl.item()

                if i % log_every == 0 and i > 0:
                    train_loss /= log_every
                    train_mse /= log_every
                    train_kl /= log_every
                    if not bar_log:
                        print(
                            '[{:d}/{:d}] MSE: {:.6f}  KL: {:.6f}  Total: {:.6f}  MSE-Norm: {:.6f}  KL-Norm: {:.6f} Time: {:.6f}'.format(
                                i, len(dataset), train_mse, train_kl, train_loss, train_mse / alpha,
                                                                                  train_kl / beta,
                                                                                  time.time() - start_time))
                    train_loss = 0
                    train_mse = 0
                    train_kl = 0

                if i % save_images_every == 0:
                    err_imgs = torch.abs(goal_batch.cpu() - out.cpu().detach())
                    Trainer.save_img_examples(d.cpu(), f'original', epoch, i)
                    Trainer.save_img_examples(out.cpu().detach(), f'reconstruction', epoch, i)
                    Trainer.save_img_examples(err_imgs, f'error', epoch, i)

                    if entanglement_data is not None:
                        Trainer.save_entanglement_img(entanglement_data, model, epoch, i)

            if epoch % save_every == 0:
                torch.save(model, f"{model_folder}{model_name}.pt")
                # torch.save(model, f"{model_folder}{model_name}_{epoch}.pt")

        torch.save(model, f"{model_folder}{model_name}.pt")


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True,
                    help="the path to the vae training data file, e.g. './data/x.npy'")
    ap.add_argument("-n", "--model_name", required=True,
                    help="the base name of the resulting vae model file, e.g. 'front_side'")
    ap.add_argument("-m", "--mode", required=True, help="multicam vae mode: 'ce', 'ec', or 'ece'")
    ap.add_argument("-e", "--epochs", required=True, help="total number of epochs to train", type=int)
    ap.add_argument("-c", "--num_cams", required=True, help="number of cameras in the training dataset", type=int)
    ap.add_argument("-l", "--latent_dim", required=True, help="number of latent dimensions", type=int)
    ap.add_argument("-b", "--batch_size", required=False, help="the batch size", type=int, default=128)
    ap.add_argument("-f", "--lefe", required=False, help="use long exposure feature extraction vae mode", type=bool,
                    default=False)
    ap.add_argument("-r", "--dataset_limit", required=False, help="number of dataset entries to use", type=int,
                    default=-1)
    ap.add_argument("-s", "--skip_outer", required=False, help="skip outer layer training for ece mode", type=bool,
                    default=False)
    args = vars(ap.parse_args())

    mode = args["mode"]
    num_cams = args["num_cams"]
    latent_dim = args["latent_dim"]
    lefe = args["lefe"]

    print("Loading dataset...")
    dataset = np.load(args["dataset"])[:, 0:num_cams]
    lefe_dataset = np.load(args["dataset"].replace("lefe_train_data", "lefe_data"))[:, 0:num_cams] if lefe else None

    width = dataset.shape[3]
    height = dataset.shape[2]

    print("Building VAE...")
    from multicamvae import ConcatEncodeVae, EncodeConcatVae, EncodeConcatEncodeVae

    if mode == "ce":
        vae = ConcatEncodeVae(num_cams, width, height, latent_dim)
    elif mode == "ec":
        vae = EncodeConcatVae(num_cams, width, height, latent_dim)
    elif mode == "ece":
        vae = EncodeConcatEncodeVae(num_cams, width, height, latent_dim)
    else:
        print("Unknown multicam vae mode '{}'.".format(mode))
        exit(-1)

    name = "mvae_model_{}_{}_{}".format(args["model_name"],
                                        (width if width == height else "{}_{}".format(width, height)),
                                        mode)

    ep = args["epochs"]
    limit = args["dataset_limit"]
    bat = args["batch_size"]

    fac = 10

    for h in range(1) if limit == -1 else tqdm(range(dataset.shape[0] * fac // limit)):
        if limit == -1:
            data = dataset
            lefe_data = lefe_dataset
        else:
            permut = np.random.permutation(dataset.shape[0])
            dataset = dataset[permut]
            if lefe:
                lefe_dataset = lefe_dataset[permut]

            data = dataset[0:limit]
            lefe_data = None if not lefe else lefe_dataset[0:limit]

            ep = args["epochs"] // fac

        if args["skip_outer"]:
            vae.train(data, model_name=name, batch_size=bat, epochs=ep, lefe=lefe, lefe_dataset=lefe_data,
                      skip_outer=True)
        else:
            vae.train(data, model_name=name, batch_size=bat, epochs=ep, lefe=lefe, lefe_dataset=lefe_data)
