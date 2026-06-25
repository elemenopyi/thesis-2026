import torch
import torchaudio
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataset import StimuliSounds
from structure import SimpleLSTM

CSV_FILE = "dataset.csv"
AUDIO_DIR = "sound/"
SAMPLES = 6394

mel_transform = torch.nn.Sequential(
    torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=1024, win_length=320,
        hop_length=44, window_fn=torch.hamming_window, n_mels=128
    ),
    torchaudio.transforms.AmplitudeToDB()
)

## load pytorch file
model = SimpleLSTM(input_size=128, hidden_size=64)
model.load_state_dict(torch.load("model.pt"))

## load from lightning logs
# model = SimpleLSTM.load_from_checkpoint("lightning_logs/version_55/checkpoints/epoch=199-step=40400.ckpt")
# model = model.to('cpu')

model.eval()

dataset = StimuliSounds(CSV_FILE, AUDIO_DIR, SAMPLES, mel_transform)
csv = pd.read_csv(CSV_FILE)
unique_files = csv.iloc[:, 0].unique()

results = []
with torch.no_grad():
    for fname in unique_files:
        idx = csv[csv.iloc[:, 0] == fname].index[0]
        mel, _ = dataset[idx]
        mel = mel.unsqueeze(0)    # (1, time, 128)
        pred = model.forward(mel).item()
        results.append((fname, pred))

res_df = pd.DataFrame(results, columns=['audio_file', 'predicted'])

def parse_filename(fname):
    parts = fname.split('_')
    vot = int(parts[2])                # step
    accent = parts[1].split('-')[0]    # asm or amr
    continuum = int(parts[0])
    accent_map = {'amr': 'American', 'asm': 'Assamese'}
    return vot, accent_map[accent], continuum

res_df[['vot', 'english', 'continuum']] = res_df['audio_file'].apply(
    lambda x: pd.Series(parse_filename(x))
)

mean_preds = res_df.groupby(['vot', 'english', 'continuum'])['predicted'].mean().reset_index()

g = sns.FacetGrid(mean_preds, col='continuum', hue='english',
                  palette={'American':'steelblue', 'Assamese':'coral'})
g.map(plt.plot, 'vot', 'predicted', marker='o')
g.map(plt.scatter, 'vot', 'predicted')
g.add_legend()
g.set_ylabels('Predicted rating')
g.set_xlabels('VOT step')
plt.show()