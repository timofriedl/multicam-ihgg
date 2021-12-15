import numpy as np
import shutil
import sys
import time
import torch
from argparse import ArgumentParser
from matplotlib import pyplot as plt
from pathlib import Path
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append("..")

from vae.innervae import InnerVae
from vae.utils import save_examples, loss_fn_weighted, loss_fn_weighted2, device

images_path = "images"


class Trainer:
    """
    Class for VAE training.
    """

    @staticmethod
    def create_images_folder():
        """
        Creates the /images folder and clears the content
        """
        path = Path(images_path)
        if path.exists():
            shutil.rmtree(images_path)
        path.mkdir()

    @staticmethod
    def save_img_examples(images: Tensor, name: str, epoch: int, it: int):
        """
        Saves a batch of images to one image file

        :param images: the pytorch tensor of images with shape [batch_size, 3, height, width]
        :param name: the name of the image file
        :param epoch: the current epoch (used for the file name)
        :param it: the current iteration (used for the file name)
        """
        save_examples(images, f'{name}', img_format="png")
        save_examples(images, f'{name}_{epoch}_{it}')

    @staticmethod
    def train_vae(dataset: DataLoader, model: nn.Module, model_name: str, goal_dataset: DataLoader = None,
                  model_folder="./models/", alpha=10., beta=1., bar_log=False, log_every=10,
                  load=True, save_images_every=1000, save_every=10, start_epoch=0, epochs=500):
        """
        Trains a given VAE

        :param dataset: the DataLoader that contains the training data
        :param model: the VAE model to train
        :param model_name: the name of the model to train, e.g. "mvae_model_fetch_reach_front_side_64_ec_0"
        :param goal_dataset: a DataLoader that contains the expected output data or None if LEFE is not used
        :param model_folder: the folder path where the models are stored
        :param alpha: the VAE alpha hyperparameter
        :param beta: the VAE beta hyperparameter
        :param bar_log: True if a progress bar should be used as console output,
                        False if default training information should be printed
        :param log_every: the interval, how often the log should be written
        :param load: True if training should be continued from latest existing model,
                     False if previously trained models should be deleted
        :param save_images_every: the interval, how often output images should be produced
        :param save_every: the interval, how often the model should be saved
        :param start_epoch: the number of the epoch to start
        :param epochs: the total number of epochs to train
        """
        if not bar_log:
            print(f'Device: {device}')

        if load:
            try:
                # Load latest model
                model = torch.load(f"{model_folder}{model_name}.pt")
            except FileNotFoundError:
                Trainer.create_images_folder()
                print("Found no model to load. Created new.")

        # If LEFE is not used, use input data set as output data set
        if goal_dataset == None:
            goal_dataset = dataset

        # Load model and create optimizer
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        # Iterate through epochs
        epoch_range = range(start_epoch + 1, epochs + 1)
        for epoch in tqdm(epoch_range) if bar_log else epoch_range:
            if not bar_log:
                print(f'Epoch [{epoch}/{epochs}]')
            start_time = time.time()

            train_loss = 0
            train_mse = 0
            train_kl = 0

            # Iterate through dataset
            for i, (d, goal_batch) in enumerate(zip(dataset, goal_dataset)):
                optimizer.zero_grad()

                # Encode data
                out, mu, logvar = model(d)

                # Apply loss function
                if model_name == 'bdvae_partial_reconstruction' or model_name == 'bdvae_full_reconstruction':
                    mse_loss, kl, loss = loss_fn_weighted(goal_batch, out, mu, logvar, alpha=alpha, beta=beta)
                else:
                    mse_loss, kl, loss = loss_fn_weighted2(goal_batch, out, mu, logvar, alpha=alpha, beta=beta)

                # Optimize
                loss.backward(retain_graph=isinstance(model, InnerVae))
                optimizer.step()

                train_loss += loss.item()
                train_mse += mse_loss.item()
                train_kl += kl.item()

                if i % log_every == 0 and i > 0:
                    # Statistics
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
                    # Output images
                    err_imgs = torch.abs(goal_batch.cpu() - out.cpu().detach())
                    Trainer.save_img_examples(d.cpu(), f'original', epoch, i)
                    Trainer.save_img_examples(out.cpu().detach(), f'reconstruction', epoch, i)
                    Trainer.save_img_examples(err_imgs, f'error', epoch, i)

            if epoch % save_every == 0:
                # Save
                torch.save(model, f"{model_folder}{model_name}.pt")
                # torch.save(model, f"{model_folder}{model_name}_{epoch}.pt")

        # Finally save again
        torch.save(model, f"{model_folder}{model_name}.pt")


if __name__ == "__main__":
    """
    python trainer.py
    
    This code can be executed to perform VAE training. For usage see the following arguments
    """
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

    # Load dataset
    print("Loading dataset...")
    dataset = np.load(args["dataset"])[:, 0:num_cams]
    lefe_dataset = np.load(args["dataset"].replace("lefe_train_data", "lefe_data"))[:, 0:num_cams] if lefe else None

    width = dataset.shape[3]
    height = dataset.shape[2]

    # Create MultiCamVae to train
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

    # Iterate through hyper batches to prevent too high memory usage
    for h in range(1) if limit == -1 else tqdm(range(dataset.shape[0] * fac // limit)):
        if limit == -1:
            data = dataset
            lefe_data = lefe_dataset
        else:
            # Choose random part of training dataset
            permut = np.random.permutation(dataset.shape[0])
            dataset = dataset[permut]
            if lefe:  # If LEFE is activated, also choose same random part of the output dataset
                lefe_dataset = lefe_dataset[permut]

            # Limit the dataset
            data = dataset[0:limit]
            lefe_data = None if not lefe else lefe_dataset[0:limit]

            ep = args["epochs"] // fac

        # Train
        if args["skip_outer"]:
            vae.train(data, model_name=name, batch_size=bat, epochs=ep, lefe=lefe, lefe_dataset=lefe_data,
                      skip_outer=True)
        else:
            vae.train(data, model_name=name, batch_size=bat, epochs=ep, lefe=lefe, lefe_dataset=lefe_data)
