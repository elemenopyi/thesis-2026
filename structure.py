# structure.py
import torch
import torch.nn as nn
import lightning as L
from torch.optim.lr_scheduler import ReduceLROnPlateau

class SimpleLSTM(L.LightningModule):
    def __init__(self, input_size=128, hidden_size=64, learning_rate=0.01):
        super().__init__()
        self.learning_rate = learning_rate
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        lstm_out, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.float().unsqueeze(1)
        pred = self.forward(x)
        loss = self.loss_fn(pred, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.float().unsqueeze(1)
        pred = self.forward(x)
        loss = self.loss_fn(pred, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }