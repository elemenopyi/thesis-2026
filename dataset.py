# dataset.py
import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import torchaudio
import librosa

class StimuliSounds(Dataset):

    def __init__(self, csv_file, audio_dir, samples, transformation):
        # Dataset class for loading stimuli and ratings
        self.csv = pd.read_csv(csv_file)
        self.audio_dir = audio_dir
        self.samples = samples  # max number of samples
        self.transformation = transformation
        # self.min_val, self.max_val = self._compute_min_max() # compute global min and max for MinMax scalling
        self.mean, self.std = self._compute_mean_std()

    def __len__(self):
        # total number of trials
        return len(self.csv)

    def __getitem__(self, index):
        # called when accessing dataset[index]
        path = os.path.join(self.audio_dir, self.csv.iloc[index, 0] + ".wav")
        response = self.csv.iloc[index, 1]

        signal, _ = librosa.load(path, sr=16000, mono=True)
        signal = torch.tensor(signal, dtype=torch.float32).unsqueeze(0) # convert to tensor

        # Force exactly self.samples
        if signal.shape[1] < self.samples:
            signal = nn.functional.pad(signal, (0, self.samples - signal.shape[1]))
        # elif signal.shape[1] > self.samples:
        #     signal = signal[:, :self.samples]

        mel = self.transformation(signal)   # (1, n_mels, time)
        # mel = (mel - self.min_val) / (self.max_val - self.min_val)    # scalling
        # Per‑band Z‑score normalisation
        mel = (mel - self.mean[:, None]) / (self.std[:, None] + 1e-8)
        mel = mel.squeeze(0).permute(1, 0)  # shape becomes (time, n_mels)
        return mel, torch.tensor(response, dtype=torch.float32)
    
    def _compute_mean_std(self):
        # We need to collect a large number of frames across all files to estimate band‑wise mean/std
        sum_val = None
        sum_sq = None
        n_frames = 0

        for i in range(len(self.csv)):
            path = os.path.join(self.audio_dir, self.csv.iloc[i, 0] + ".wav")
            sig, _ = librosa.load(path, sr=16000, mono=True)
            sig = torch.tensor(sig, dtype=torch.float32).unsqueeze(0)
            if sig.shape[1] < self.samples:
                sig = torch.nn.functional.pad(sig, (0, self.samples - sig.shape[1]))
            elif sig.shape[1] > self.samples:
                sig = sig[:, :self.samples]

            mel = self.transformation(sig)          # (1, 128, time)
            mel = mel.squeeze(0)                    # (128, time)
            n_frames += mel.shape[1]

            if sum_val is None:
                sum_val = mel.sum(dim=1)            # (128,)
                sum_sq = (mel ** 2).sum(dim=1)
            else:
                sum_val += mel.sum(dim=1)
                sum_sq += (mel ** 2).sum(dim=1)

        mean = sum_val / n_frames
        std = torch.sqrt(sum_sq / n_frames - mean ** 2)
        return mean, std

    # def _compute_min_max(self):
    #     min_val = float('inf')
    #     max_val = float('-inf')
    #     for i in range(len(self.csv)):
    #         path = os.path.join(self.audio_dir, self.csv.iloc[i, 0] + ".wav")
    #         sig, _ = librosa.load(path, sr=16000, mono=True)
    #         sig = torch.tensor(sig, dtype=torch.float32).unsqueeze(0)
    #         if sig.shape[1] < self.samples:
    #             sig = nn.functional.pad(sig, (0, self.samples - sig.shape[1]))
    #         elif sig.shape[1] > self.samples:
    #             sig = sig[:, :self.samples]
    #         mel = self.transformation(sig)
    #         min_val = min(min_val, mel.min().item())
    #         max_val = max(max_val, mel.max().item())
    #     return min_val, max_val

mel_transform = torch.nn.Sequential(
    torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=1024,
        win_length=320,
        hop_length=44,
        window_fn=torch.hamming_window,
        n_mels=128
    ),
    torchaudio.transforms.AmplitudeToDB()
)    

dataset = StimuliSounds("dataset.csv", "sound/", 6394, mel_transform)
mel0, _ = dataset[0]
mel1, _ = dataset[1]
print("Shape 0:", mel0.shape)   # should be (time_frames, 128)
print("Shape 1:", mel1.shape)   # should be identical to the first

# mel, _ = dataset[0]
# print("Actual time frames:", mel.shape[0])   # (time, 128)
# effective_length = 6394 + 1024   # 6394 + 1024 = 7418
# num_frames = 1 + (effective_length - 320) // 44
# print("Predicted frames:", num_frames)
# print(mel_transform[0].n_fft, mel_transform[0].center)