import abc
import numpy as np
import sys
import torch
from math import ceil
from torch import Tensor
from torch.utils.data import DataLoader
from vae.coordbdvae import CoordBDVAE
from vae.innervae import InnerVae
from vae.trainer import Trainer
from vae.utils import image_to_tensor, tensor_to_image, images_to_tensor, tensor_wrap, \
    images_to_tensor_cpu, device


class MultiCamVae:
    """
    An implementation of CoordBDVAE for Multi-Camera input.
    """

    def __init__(self, num_cams: int, width: int, height: int, latent_dim: int):
        """
        Creates a new MultiCamVae instance.

        :param num_cams: the number of cameras that deliver input images
        :param width: the horizontal image size of each camera
        :param height: the vertical image size of each camera
        :param latent_dim: the number of latent dimensions
        """
        self.num_cams = num_cams
        self.width = width
        self.height = height
        self.latent_dim = latent_dim

    @abc.abstractmethod
    def encode(self, images: np.ndarray) -> Tensor:
        """
        Encodes a given vector of images to its corresponding latent representation

        :param images: the numpy array of images to encode with shape [num_cams, height, width, 3]
        :return: a pytorch tensor with shape [latent_dim] that represents compressed latent vector
        """
        return

    @abc.abstractmethod
    def decode(self, latent: Tensor) -> np.ndarray:
        """
        Decodes a given latent vector to reconstructions of the original images

        :param latent: the pytorch tensor with shape [latent_dim] that represents the latent vector
        :return: a numpy array of decoded reconstruction images
                 with shape [num_cams, height, width, 3]
        """
        return

    def forward(self, images: np.ndarray) -> np.ndarray:
        """
        Encodes a given array of images to its latent representation
        and decodes it back to reconstructions of the original images

        :param images: the numpy array of images with shape [num_cams, height, width, 3]
        :return: a numpy array of reconstruction images with same shape
        """
        return self.decode(self.encode(images))

    @abc.abstractmethod
    def train(self, dataset: np.ndarray, model_name: str, batch_size: int, epochs: int, lefe: bool,
              lefe_dataset: np.ndarray):
        """
        Trains this MultiCamVae

        :param dataset: the numpy array of input image vectors with shape [dataset_size, num_cams, height, width, 3]
        :param model_name: the name of the model to train, e.g. "mvae_model_fetch_reach_front_side_64_ec"
        :param batch_size: the VAE training batch size
        :param epochs: the number of epochs to train the VAE
        :param lefe: a flag if Long Exposure Feature Extraction is being used
        :param lefe_dataset: a second numpy array of input image vectors with same shape as the input dataset
                             that is used as the dataset of correct reconstruction image vectors.
                             If lefe == False, None can be used as lefe_dataset
        """
        return

    @staticmethod
    @abc.abstractmethod
    def load(base_path: str, num_cams: int, width: int, height: int, latent_dim: int):
        """
        Loads this MultiCamVae with the specified properties

        :param base_path: the base path of the model to load, e.g. "mvae_model_fetch_reach_front_side_64_ec"
        :param num_cams: the number of cameras
        :param width: the horizontal image size
        :param height: the vertical image size
        :latent_dim: the number of latent dimensions
        :return: the loaded MultiCamVae
        """
        return


class ConcatEncodeVae(MultiCamVae):
    """
    A ConcatEncodeVae is a MultiCamVae that first concatenates all input images,
    and then compresses them to one single latent vector
    """

    def __init__(self, num_cams: int, width: int, height: int, latent_dim: int):
        """
        Creates a new ConcatEncodeVae instance. For parameter list see MultiCamVae

        ConcatEncodeVae consists of one single CoordBDVAE that is used to compress concatenated input images
        """
        super().__init__(num_cams, width, height, latent_dim)
        self.vae = CoordBDVAE(width * num_cams, height, latent_dim)
        self.inner_lat_dim = latent_dim

    def encode(self, images: np.ndarray) -> Tensor:
        """
        Implements MultiCamVae.encode

        Images are first concatenated horizontally, then encoded and sampled using CoordBDVAE
        """
        single_image = np.concatenate(images, axis=1)
        mu, logvar = self.vae.encode(image_to_tensor(single_image))
        return self.vae.sample(mu, logvar)[0]

    def decode(self, latent: Tensor) -> np.ndarray:
        """
        Implements MultiCamVae.decode

        Latent vectors are first decoded using CoordBDVAE, then split back into separate images
        """
        concat_image = tensor_to_image(self.vae.decode(tensor_wrap(latent)))
        return np.array(np.hsplit(concat_image, self.num_cams))

    def train(self, dataset: np.ndarray, model_name: str, batch_size: int, epochs: int, lefe=False, lefe_dataset=None):
        """
        Implements MultiCamVae.train

        For the training of ConcatEncodeVae, a concatenated dataset is generated.
        Then the underlying CoordBDVAE is trained.
        """

        # Concatenate images
        concat_dataset = np.empty([dataset.shape[0], self.vae.height, self.vae.width, 3])
        if lefe:
            concat_lefe_dataset = np.empty(concat_dataset.shape)

        for i in range(dataset.shape[0]):
            concat_dataset[i] = np.concatenate(dataset[i, :self.num_cams], axis=1)
            if lefe:  # If LEFE mode is activated, second "output" data set has to be created
                concat_lefe_dataset[i] = np.concatenate(lefe_dataset[i, :self.num_cams], axis=1)

        # Convert to tensor
        data = images_to_tensor(concat_dataset)
        dl = DataLoader(data, batch_size=batch_size, shuffle=not lefe)
        if lefe:
            lefe_data = images_to_tensor(concat_lefe_dataset)
            lefe_dl = DataLoader(lefe_data, batch_size=batch_size, shuffle=False)

        # Do training
        beta = 10. if "pick_and_place" in model_name else 1.
        Trainer.train_vae(dl, self.vae, model_name, epochs=epochs, bar_log=True, goal_dataset=lefe_dl if lefe else dl,
                          beta=beta)

    @staticmethod
    def load(base_path: str, num_cams: int):
        """
        Implements MultiCamVae.load
        """
        coordbdvae = torch.load(base_path + ".pt")
        mvae = ConcatEncodeVae(num_cams, coordbdvae.width // num_cams, coordbdvae.height, coordbdvae.latent_dim)
        mvae.vae = coordbdvae
        return mvae


class EncodeConcatVae(MultiCamVae):
    """
    An EncodeConvatVae is a MultiCamVae that first encodes input images separately,
    and then concatenates the latent vectors to one single large latent vector
    """

    def __init__(self, num_cams: int, width: int, height: int, latent_dim: int):
        """
        Creates a new ConcatEncodeVae instance. For parameter list see MultiCamVae

        EncodeConcatVae consists of multiple CoordBDVAEs, one for each camera
        """
        super().__init__(num_cams, width, height, latent_dim)
        self.vaes = list(map(lambda _: CoordBDVAE(width, height, latent_dim), range(num_cams)))
        self.inner_lat_dim = latent_dim * self.num_cams

    def encode(self, images: np.ndarray) -> Tensor:
        """
        Implements MultiCamVae.encode

        Images are first encoded separately, and then concatenated to one single latent vector
        """
        images = images_to_tensor(images)

        latent = torch.empty(self.inner_lat_dim)
        for c in range(self.num_cams):  # Iterate through all cameras and collect latent vectors
            mu, logvar = self.vaes[c].encode(tensor_wrap(images[c]))
            z = self.vaes[c].sample(mu, logvar)[0]
            latent[(c * self.latent_dim):((c + 1) * self.latent_dim)] = z

        return latent

    def decode(self, latent: Tensor) -> np.ndarray:
        """
        Implements MultiCamVae.decode

        The latent vector is first split into latent vectors for each camera,
        then the images are reconstructed one-by-one for each camera
        """
        latent = latent.reshape(self.num_cams, self.latent_dim).to(device)

        images = np.empty([self.num_cams, self.height, self.width, 3], dtype=np.uint8)
        for c in range(self.num_cams):  # Iterate through all cameras and decode the corresponding latent vector
            images[c] = tensor_to_image(self.vaes[c].decode(tensor_wrap(latent[c])))

        return images

    def train(self, dataset: np.ndarray, model_name: str, batch_size: int, epochs: int, lefe=False, lefe_dataset=None):
        """
        Implements MultiCamVae.train

        For the training of EncodeConcatVae, the CoordBDVAEs are trained one-by-one.
        """
        for c in range(self.num_cams):  # For each camera perspective train CoordBDVAE
            print("VAE %d/%d:" % (c + 1, self.num_cams))

            # Initialize dataset
            data = images_to_tensor(dataset[:, c])
            dl = DataLoader(data, batch_size=batch_size, shuffle=not lefe)

            if lefe:  # If LEFE mode is used, also generate an output data set
                lefe_data = images_to_tensor(lefe_dataset[:, c])
                lefe_dl = DataLoader(lefe_data, batch_size=batch_size, shuffle=False)

            # Do training
            beta = 10. if "pick_and_place" in model_name else 1.
            Trainer.train_vae(dl, self.vaes[c], model_name="{}_{}".format(model_name, c),
                              goal_dataset=lefe_dl if lefe else dl,
                              epochs=epochs // self.num_cams, bar_log=True, beta=beta)

    @staticmethod
    def load(base_path: str, num_cams: int):
        """
        Implements MultiCamVae.load

        CoordBDVAEs are loaded separately, then combined to one new EncodeConcatVae instance
        """
        vaes = list()
        for i in range(num_cams):
            vaes.append(torch.load("{}_{}.pt".format(base_path, i)))

        mvae = EncodeConcatVae(num_cams, vaes[0].width, vaes[0].height, vaes[0].latent_dim)
        mvae.vaes = vaes
        return mvae


class EncodeConcatEncodeVae(MultiCamVae):
    """
    An EncodeConcatEncodeVae is an extended version of EncodeConcatVae where images are separately compressed,
    then the latent vectors are combined to one large latent vector,
    then this vector is being encoded in an inner stage again using an InnerVae
    """

    def __init__(self, num_cams: int, width: int, height: int, latent_dim: int):
        """
        Creates a new ConcatEncodeVae instance. For parameter list see MultiCamVae

        In addition to the CoordBDVAEs, another InnerVae is being used for inner stage compression
        """
        super().__init__(num_cams, width, height, latent_dim)
        self.vaes = list(map(lambda _: CoordBDVAE(width, height, latent_dim), range(num_cams)))
        self.inner_vae = InnerVae(input_dim=num_cams * latent_dim, latent_dim=latent_dim)
        self.inner_lat_dim = latent_dim

    def outer_encode(self, images: np.ndarray) -> Tensor:
        """
        Helper method for outer stage compression. Images are compressed separately,
        then the latent vectors are being concatenated.

        :param images: the numpy array of images with shape [num_cams, height, width, 3]
                       to compress to an outer latent vector
        :return: a tensor with shape [num_cams * latent_dim] containing the latent vector entries
        """
        images = images_to_tensor_cpu(images)

        outer_latent = torch.empty(self.num_cams * self.latent_dim)
        for c in range(self.num_cams):  # Iterate through all cameras and encode images separately
            model = self.vaes[c].to(device)
            mu, logvar = model.encode(tensor_wrap(images[c].to(device)))
            z = model.sample(mu, logvar)[0]

            outer_latent[(c * self.latent_dim):((c + 1) * self.latent_dim)] = z

        return outer_latent

    def outer_encode_all(self, images: np.ndarray) -> Tensor:
        """
        Helper method to perform outer stage compression with a batch of image vectors.

        :param images: the numpy array of image vectors with shape [batch_size, num_cams, height, width, 3]
        :return: a tensor with shape [batch_size, num_cams * latent_dim] containing the latent vectors
        """
        images = images_to_tensor_cpu(images.reshape(images.shape[0] * self.num_cams, self.height, self.width, 3)) \
            .reshape(images.shape[0], self.num_cams, 3, self.height, self.width)

        outer_latent = torch.empty(images.shape[0], self.num_cams * self.latent_dim)
        for c in range(self.num_cams):  # Iterate through all cameras and encode whole batch of images in one step
            model = self.vaes[c].to(device)
            mu, logvar = model.encode(images[:, c].to(device))
            z = model.sample(mu, logvar)

            min_pos = c * self.latent_dim
            max_pos = (c + 1) * self.latent_dim
            outer_latent[:, min_pos:max_pos] = z

        return outer_latent

    def inner_encode(self, outer_latent: Tensor) -> Tensor:
        """
        Helper method for inner stage compression.
        InnerVae is used to encode outer stage latent vector to inner stage latent vector

        :param outer_latent: the pytorch tensor that contains the outer latent vector entries
        :return: a tensor with shape [latent_dim] containing the inner latent vector entries
        """
        model = self.inner_vae.to(device)
        mu, logvar = model.encode(tensor_wrap(outer_latent.to(device)))
        return model.sample(mu, logvar)[0]

    def encode(self, images: np.ndarray) -> Tensor:
        """
        Implements MultiCamVae.encode

        First outer stage compression, then inner stage compression
        """
        return self.inner_encode(self.outer_encode(images))

    def inner_decode(self, inner_latent: Tensor) -> Tensor:
        """
        Helper method for inner stage decoding.
        InnerVae is used to decode inner stage latent vector to outer stage latent vector

        :param inner_latent: the pytorch tensor that contains the inner latent vector entries
        :return: a tensor with shape [latent_dim] containing the outer latent vector entries
        """
        return self.inner_vae.decode(tensor_wrap(inner_latent))[0]

    def outer_decode(self, outer_latent: Tensor) -> np.ndarray:
        """
        Helper method for outer stage decoding.

        CoordBDVAEs are used to decode the corresponding parts of the outer latent vector
        back to reconstructions of the input images
        """
        latent = outer_latent.reshape(self.num_cams, self.latent_dim)

        images = np.empty([self.num_cams, self.height, self.width, 3], dtype=np.uint8)
        for c in range(self.num_cams):  # Iterate through cameras and decode images one-by-one
            images[c] = tensor_to_image(self.vaes[c].decode(tensor_wrap(latent[c])))

        return images

    def decode(self, latent: Tensor) -> np.ndarray:
        """
        Implements MultiCamVae.decode

        First inner stage is decoded, then outer stage
        """
        return self.outer_decode(self.inner_decode(latent))

    def train(self, dataset: np.ndarray, model_name: str, batch_size: int, epochs: int, lefe: bool,
              lefe_dataset: np.ndarray, skip_outer=False):
        """
        Implements MultiCamVae.train

        For the training of EncodeConcatEncodeVae, first the CoordBDVAEs are trained one-by-one.
        Then another InnerVae is trained to perform the inner stage compression
        """
        for c in range(self.num_cams):  # Iterate through all cameras and perform training of outer stage
            print("VAE %d/%d:" % (c + 1, self.num_cams + 1))

            # Initialize dataset
            data = images_to_tensor(dataset[:, c])
            dl = DataLoader(data, batch_size=batch_size, shuffle=not lefe)

            if lefe:  # If LEFE is activated, load another "output" data set
                lefe_data = images_to_tensor(lefe_dataset[:, c])
                lefe_dl = DataLoader(lefe_data, batch_size=batch_size, shuffle=False)

            # Do training
            ep = 0 if skip_outer else epochs // (self.num_cams + 1)
            beta = 10. if "pick_and_place" in model_name else 1.
            Trainer.train_vae(dl, self.vaes[c], model_name="{}_{}".format(model_name, c), epochs=ep, bar_log=True,
                              goal_dataset=lefe_dl if lefe else dl, beta=beta)

        # Now perform the inner stage training
        print("Inner VAE:")
        data = torch.empty(dataset.shape[0], self.num_cams * self.latent_dim, dtype=torch.float32)
        for b in range(ceil(data.shape[0] / batch_size)):  # Split inner data set into batches
            pos_min = b * batch_size
            pos_max = min(dataset.shape[0], (b + 1) * batch_size)
            # Now encode images to outer latent vectors
            data[pos_min:pos_max] = self.outer_encode_all(dataset[pos_min:pos_max])

        # Create DataLoader from generated outer latent data set
        dl = DataLoader(data.to(device), batch_size=batch_size, shuffle=True)

        # Do training
        ep = epochs // (self.num_cams + 1)
        Trainer.train_vae(dl, self.inner_vae, model_name="{}_inner".format(model_name), epochs=ep, bar_log=True,
                          beta=5., save_images_every=sys.maxsize)

    @staticmethod
    def load(base_path: str, num_cams: int):
        """
        Implements MultiCamVae.load

        For EncodeConcatEncodeVae, CoordBDVAEs are loaded like in EncodeConcatVae.load.
        Additionally, InnerVae is loaded.
        """
        vaes = list()
        for i in range(num_cams):
            vaes.append(torch.load("{}_{}.pt".format(base_path, i)))

        inner_vae = torch.load("{}_inner.pt".format(base_path))

        mvae = EncodeConcatEncodeVae(num_cams, vaes[0].width, vaes[0].height, vaes[0].latent_dim)
        mvae.vaes = vaes
        mvae.inner_vae = inner_vae
        return mvae
