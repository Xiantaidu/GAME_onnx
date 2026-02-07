import random

import numpy as np
import torch

# {'NoiseType': 1, 'probability': 0.1,
#  'NoiseArgs': [{'name': 'xxx', 'value': 0.2, 'min': 0.02, 'max': 0.1, 'UseRandomValue': True}]}


def add_gaussian_noise(tensor, mean=0, std=1, noise_level=0.001):
    noise = torch.randn(tensor.size()) * std + mean
    noisy_tensor = tensor + noise * noise_level
    return noisy_tensor


def add_uniform_noise(tensor, low=-1, high=1, noise_level=0.01):
    noise = torch.rand(tensor.size()) * (high - low) + low
    noisy_tensor = tensor + noise * noise_level
    return noisy_tensor


def add_poisson_noise(tensor, lam=1, noise_level=0.01):
    noise = torch.poisson(lam * torch.ones(tensor.size()))
    noisy_tensor = tensor + noise * noise_level
    return noisy_tensor


def add_salt_and_pepper_noise(tensor, salt_prob=0.5, pepper_prob=0.5):
    total_pixels = tensor.numel()
    num_salt = int(total_pixels * salt_prob + 1)
    num_pepper = int(total_pixels * pepper_prob + 1)

    # Add salt noise
    salt_indices = torch.randint(low=0, high=total_pixels, size=(num_salt,))
    tensor.view(-1)[salt_indices] = 1

    # Add pepper noise
    pepper_indices = torch.randint(low=0, high=total_pixels, size=(num_pepper,))
    tensor.view(-1)[pepper_indices] = 0

    return tensor


def add_impulse_noise(signal, num_impulses, impulse_strength):
    impulse_indices = torch.randint(low=0, high=len(signal), size=(num_impulses,))
    noisy_signal = signal.clone()
    noisy_signal[impulse_indices] += impulse_strength
    return noisy_signal


def add_multiplicative_noise(signal, noise_level=0.01):
    noise = torch.randn_like(signal)
    noisy_signal = signal * (1 + noise_level * noise)
    return noisy_signal


def pink_noise(N):
    X_white = np.fft.rfft(np.random.randn(N + 1))

    f = np.fft.rfftfreq(N + 1)

    S = 1 / np.where(f == 0, float('inf'), np.sqrt(f))

    S = S / np.sqrt(np.mean(S ** 2))

    X_shaped = X_white * S
    return np.fft.irfft(X_shaped)[:N]


def add_pink_noise(signal, noise_level=0.001):
    noise = pink_noise(len(signal))

    noisy_signal = signal + noise_level * torch.from_numpy(noise).float()
    return noisy_signal


class DataAugNoise:
    def __init__(self, config):
        self.add_probability = config['noise_aug']['use_noise_aug_probability']
        self.noise_cfg = config['noise_aug']['noise_config']
        self.noise_diet = {'gaussian': add_gaussian_noise, 'impulse_noise': add_impulse_noise,
                           'uniform_noise': add_uniform_noise, 'poisson_noise': add_poisson_noise,
                           'salt_and_pepper_noise': add_salt_and_pepper_noise,
                           'multiplicative_noise': add_multiplicative_noise, 'pink_noise': add_pink_noise,
                           }

    def get_random_noise_level(self, mins, maxs):
        return random.uniform(a=mins, b=maxs)

    def prepara_args(self, args):
        tempD = {}
        for i in args:
            if i['UseRandomValue']:
                tempD[i['name']] = self.get_random_noise_level(mins=i['min'], maxs=i['max'])
            else:
                tempD[i['name']] = i['value']
        return tempD

    @torch.no_grad()
    def add_noise(self, x):
        if random.random() < self.add_probability:
            for i in self.noise_cfg:
                if random.random() < i['probability']:
                    args = self.prepara_args(i['NoiseArgs'])
                    x = self.noise_diet[i['NoiseType']](x, **args)

        else:
            pass
        return x
