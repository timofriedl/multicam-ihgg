import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision.utils import make_grid, save_image

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def images_to_tensor(images: np.ndarray) -> Tensor:
    images = images.copy()
    images = images.reshape((images.shape[0], images.shape[1], images.shape[2], 3))
    images = images.transpose((0, 3, 1, 2)) / 255.0
    images = torch.tensor(images, dtype=torch.float32)
    images = images.to(device)
    return images


def np_to_tensor(array: np.ndarray) -> Tensor:
    return torch.tensor(array, dtype=torch.float32).to(device)


def tensor_to_np(tensor: Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def images_to_tensor_cpu(images: np.ndarray, size: int = 64) -> Tensor:
    images = images.reshape((images.shape[0], size, size, 3))
    images = images.transpose((0, 3, 1, 2)) / 255.0
    # images = torch.from_numpy(images)
    # images = images.type(torch.FloatTensor)
    images = torch.tensor(images, dtype=torch.float32)
    images = images.to('cpu')
    return images


def image_to_tensor(image: np.ndarray) -> Tensor:
    images = image.copy()
    images = images.reshape((1, image.shape[0], image.shape[1], 3))
    images = images.transpose((0, 3, 1, 2)) / 255.0
    images = torch.tensor(images, dtype=torch.float32)
    images = images.to(device)
    return images


def tensor_to_image(tensor: Tensor) -> np.ndarray:
    image = tensor.detach().cpu().numpy()
    image[(image < 0.)] = 0.
    image[(image > 1.)] = 1.
    image = np.floor(255 * image).astype(np.uint8).reshape(tensor.shape[1:])
    return image.transpose(1, 2, 0)


def save_examples(examples: Tensor, name: str, img_format: str = "jpeg") -> None:
    clipped = torch.clamp(examples.detach(), 0, 1)
    image = make_grid(clipped)
    save_image(image, "images/{0}.{1}".format(name, img_format))


def loss_fn(x: Tensor, x_rec: Tensor, mu: float, logvar: float, alpha: float = 10., beta: float = 1.) -> \
        (float, float, float):
    batch_size = x.size(0)

    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()
    mse_loss = F.mse_loss(x_rec, x, reduction='none')
    mse_loss = mse_loss.view(batch_size, -1)
    mse_loss = mse_loss.sum(dim=-1)
    mse_loss = mse_loss.mean()

    return alpha * mse_loss, beta * kl, alpha * mse_loss + beta * kl


def loss_fn_weighted(x: Tensor, x_rec: Tensor, mu: float, logvar: float, alpha: float = 10., beta: float = 1.) -> \
        (float, float, float):
    batch_size = x.size(0)

    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()
    mse_loss = F.mse_loss(x_rec, x, reduction='none')
    a = torch.ones_like(mse_loss)
    a[:, :, 40:, :] = 2.
    mse_loss = mse_loss * a
    mse_loss = mse_loss.view(batch_size, -1)
    mse_loss = mse_loss.sum(dim=-1)
    mse_loss = mse_loss.mean()

    return alpha * mse_loss, beta * kl, alpha * mse_loss + beta * kl


def loss_fn_weighted2(x: Tensor, x_rec: Tensor, mu: float, logvar: float, alpha: float = 10., beta: float = 1.) -> \
        (float, float, float):
    batch_size = x.size(0)

    if len(x.shape) == 2:
        x = x.view(batch_size, 1, 1, -1)
        x_rec = x_rec.view(batch_size, 1, 1, -1)

    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()
    mse_loss = F.mse_loss(x_rec, x, reduction='none')
    a = torch.ones_like(mse_loss)
    a[:, :, :60, :] = 2.
    mse_loss = mse_loss * a
    mse_loss = mse_loss.view(batch_size, -1)
    mse_loss = mse_loss.sum(dim=-1)
    mse_loss = mse_loss.mean()

    return alpha * mse_loss, beta * kl, alpha * mse_loss + beta * kl


def tensor_wrap(x: Tensor) -> Tensor:
    return x.unsqueeze(0)
