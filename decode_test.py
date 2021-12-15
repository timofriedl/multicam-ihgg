import numpy as np
import re
import torch
from common import get_args
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button
from tqdm import tqdm
from vae.import_vae import import_vae
from vae.multicamvae import EncodeConcatVae
from vae.utils import tensor_to_np, device

"""
This is a programm to evaluate MultiCamVae decoding performance.
"""


def update(val):
    """
    Takes the slider information as a latent vector and creates a reconstruction using the given MultiCamVae.

    :param val: unused
    """
    latent = torch.tensor(list(map(lambda s: s.val, sliders)), dtype=torch.float32)
    rec = np.concatenate(vae.decode(latent.to(device)), axis=1)
    img.set_data(rec)
    fig.canvas.draw_idle()


def reset(event):
    """
    Resets the sliders

    :param event: unused
    """
    for slider in sliders:
        slider.reset()


def rand(event):
    """
    Randomizes the latent vector

    :param event: unused
    """
    lat = vae.encode(dataset[np.random.randint(dataset.shape[0])])
    for i, slider in enumerate(sliders):
        slider.set_val(lat[i])
    rec = np.concatenate(vae.decode(lat.to(device)), axis=1)
    img.set_data(rec)
    fig.canvas.draw_idle()


if __name__ == "__main__":
    """
    decode_test.py
    
    For usage see README.md
    """
    args = get_args(clear_log=False)

    vae = import_vae(args.env, args.cams, args.mvae_mode, args.img_width, args.img_height)

    latent_dim = vae.latent_dim
    if isinstance(vae, EncodeConcatVae):
        latent_dim *= len(args.cams)

    # Create min and max values for the sliders
    min = torch.tensor([-1E10]).repeat(latent_dim).reshape(-1, 1)
    max = torch.tensor([1E10]).repeat(latent_dim).reshape(-1, 1)
    latent_range = torch.hstack((max, min)).to(device)

    # Load part of the dataset
    print("Loading dataset...")
    w = args.img_width
    h = args.img_height
    env_parts = re.findall("[A-Z][^A-Z]*", args.env.split("-")[0])
    path = "./vae/data/mvae_{}train_data_{}_{}_{}.npy" \
        .format("lefe_" if "FetchPush" in args.env or "FetchPickAndPlace" in args.env else "",
                "_".join(env_parts).lower(),
                "front_side_top",
                w if w == h else "{}_{}".format(w, h))
    dataset = np.load(path)[:8192, :2]
    np.random.shuffle(dataset)

    # Set min and max to global extrema
    print("Normalizing latent values...")
    for i in tqdm(range(dataset.shape[0] // 100)):
        latent = vae.encode(dataset[i]).to(device)
        latent_range[:, 0] = torch.minimum(latent_range[:, 0], latent)
        latent_range[:, 1] = torch.maximum(latent_range[:, 1], latent)
        torch.cuda.empty_cache()

    # Following code could be used to allow sliders to be outside [min, max]
    """
    alpha = .1
    latent_range[:, 0] -= alpha * (latent_range[:, 1] - latent_range[:, 0])
    latent_range[:, 1] += alpha * (latent_range[:, 1] - latent_range[:, 0])
    """

    # Encode one sample image vector
    sample_latent = vae.encode(dataset[0])

    # Create figure
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
