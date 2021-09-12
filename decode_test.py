import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button
from tqdm import tqdm

from common import get_args
from vae.import_vae import import_vae
from vae.multicamvae import EncodeConcatVae
from vae.utils import tensor_to_np, device


def update(val):
    latent = torch.tensor(list(map(lambda s: s.val, sliders)), dtype=torch.float32)
    rec = np.concatenate(vae.decode(latent.to(device)), axis=1)
    img.set_data(rec)
    fig.canvas.draw_idle()


def reset(event):
    for slider in sliders:
        slider.reset()


def rand(event):
    lat = vae.encode(dataset[np.random.randint(dataset.shape[0])])
    for i, slider in enumerate(sliders):
        slider.set_val(lat[i])
    rec = np.concatenate(vae.decode(lat.to(device)), axis=1)
    img.set_data(rec)
    fig.canvas.draw_idle()


if __name__ == "__main__":
    args = get_args()

    vae = import_vae(args.env, args.cams, args.mvae_mode, args.img_width, args.img_height)

    latent_dim = vae.latent_dim
    if isinstance(vae, EncodeConcatVae):
        latent_dim *= len(args.cams)

    print("Normalizing latent values...")
    min = torch.tensor([-1E10]).repeat(latent_dim).reshape(-1, 1)
    max = torch.tensor([1E10]).repeat(latent_dim).reshape(-1, 1)
    latent_range = torch.hstack((max, min)).to(device)
    dataset = np.load("./vae/data/mvae_train_data_fetch_push_front_side_top_64.npy")[:, 0:2]
    for i in tqdm(range(dataset.shape[0] // 100)):
        latent = vae.encode(dataset[i]).to(device)
        latent_range[:, 0] = torch.minimum(latent_range[:, 0], latent)
        latent_range[:, 1] = torch.maximum(latent_range[:, 1], latent)
        torch.cuda.empty_cache()

    alpha = .1
    latent_range[:, 0] -= alpha * (latent_range[:, 1] - latent_range[:, 0])
    latent_range[:, 1] += alpha * (latent_range[:, 1] - latent_range[:, 0])

    sample_latent = vae.encode(dataset[0])

    fig, ax = plt.subplots()
    fig.set_size_inches(15, 6)
    img = plt.imshow(np.concatenate(vae.decode(sample_latent), axis=1))
    plt.subplots_adjust(left=0.5)

    sliders = list()
    axamps = list()
    for d in range(latent_dim):
        axamps.append(plt.axes([0.05, 0.05 + ((1.0 - 0.05 * 2) / latent_dim) * d, 0.30, 0.05]))
        sliders.append(Slider(
            ax=axamps[-1],
            label=str(d),
            valmin=tensor_to_np(latent_range)[d, 0],
            valmax=tensor_to_np(latent_range)[d, 1],
            valinit=tensor_to_np(sample_latent)[d]
        ))
        sliders[-1].on_changed(update)

    resetax = plt.axes([0.8, 0.025, 0.1, 0.05])
    reset_button = Button(resetax, 'Reset')
    reset_button.on_clicked(reset)

    randax = plt.axes([0.65, 0.025, 0.1, 0.05])
    rand_button = Button(randax, 'Random')
    rand_button.on_clicked(rand)

    plt.show()
