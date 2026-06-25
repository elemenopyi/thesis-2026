import torch
import torchaudio
import torch.nn as nn
import lightning as L
import pandas as pd
from torch.utils.data import random_split, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset import StimuliSounds
from structure import SimpleLSTM          # 64→1 model

# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------
CSV_FILE = "dataset.csv"    # both continua
AUDIO_DIR = "sound/"
SAMPLES = 6394
BATCH_SIZE = 16
MAX_EPOCHS = 10
LEARNING_RATE = 0.01

mel_transform = nn.Sequential(
    torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=1024,
        win_length=320, #128,
        hop_length=44, #32,
        window_fn=torch.hamming_window,
        n_mels=128
    ),
    torchaudio.transforms.AmplitudeToDB()
)

# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------
dataset = StimuliSounds(CSV_FILE, AUDIO_DIR, SAMPLES, mel_transform)
csv = pd.read_csv(CSV_FILE)
n_val = int(0.2 * len(dataset))
n_train = len(dataset) - n_val
train_ds, val_ds = random_split(dataset, [n_train, n_val])
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# ------------------------------------------------------------
# Model
# ------------------------------------------------------------
model = SimpleLSTM(input_size=128, hidden_size=64, learning_rate=LEARNING_RATE)

# ------------------------------------------------------------
# Train
# ------------------------------------------------------------
trainer = L.Trainer(max_epochs=200)
trainer.fit(model, train_loader, val_loader) #, ckpt_path="lightning_logs/version_55/checkpoints/epoch=199-step=40400.ckpt")
torch.save(model.state_dict(), "model-new.pt")

# Quick sanity
model.eval()
with torch.no_grad():
    sample_loader = DataLoader(val_ds, batch_size=5, shuffle=True)
    x, y = next(iter(sample_loader))
    pred = model.forward(x).squeeze()
    print("True ratings:", y.tolist())
    print("Predicted:   ", pred.tolist())

# Test continuum 1 files
test_files = ["2_amr-eng_1.wav", "2_amr-eng_7.wav", "2_asm-eng_1.wav", "2_asm-eng_7.wav"]
for fname in test_files:
    idx = csv[csv.iloc[:, 0] == fname].index
    if len(idx) > 0:
        mel, _ = dataset[idx[0]]
        mel = mel.unsqueeze(0)
        pred = model.forward(mel).item()
        print(f"{fname:30s} predicted: {pred:.4f}")