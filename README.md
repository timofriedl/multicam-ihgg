# multicam-I-HGG

Implementation for the Bachelor's Thesis "Multi-Camera Setup in Image-Based Hindsight Goal Generation" (Timo Friedl,
2021) based on the implementation of [I-HGG](https://github.com/hakrrr/I-HGG) (James Li, 2020) and work from Aleksandar
Aleksandrov.

## Requirements

1. Ubuntu 20.04
2. Python 3.7.10
3. MuJoCo (see instructions on https://github.com/openai/mujoco-py)
4. Create a conda environment from `environment.yml`

## VAE

### Training

Train a multicam VAE model with commands like:

```
cd vae
python trainer.py -d ./data/mvae_train_data_fetch_reach_front_side_top_64.npy -n fetch_reach_front_side -m ec -e 500 -c 2 -l 8
```

with one of the multicam modes:

- ce (concat-encode)
- ec (encode-concat)
- ece (encode-concat-encode)

To train with single-cam mode choose `-m ec -c 1`.\
For more information see the usage `python trainer.py -h`.

### Decoder Evaluation

Test your trained VAE model with this script:

```
python decode_test.py
```

## HGG

### Training

Train the agent with HGG by running

```
python train.py --tag 100 --learn hgg --env=FetchReach-v1 --epoches 40 --cycles 20
```

### Playing

To generate a video looking at the agent solving the respective task according to his learned policy, issue the
following command:

```
python play_new.py --env FetchReach-v1 --play_path log/100-FetchReach-v1-hgg-ec/ --mvae_mode ec --cams front_side --play_epoch best
```

### Exporting Plot

To plot the agent's performance on multiple training runs, execute

```
python plot.py FetchReach-v1 --mvae_mode ec --cams front_side
```