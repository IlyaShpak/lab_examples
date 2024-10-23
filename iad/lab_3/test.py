import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, activation=nn.ReLU()):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(8, 6),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(6, 8),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded
model = Autoencoder()
print(model.encoder[0].weight.data.shape)