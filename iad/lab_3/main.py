import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm


class Autoencoder(nn.Module):
    def __init__(self, layer_in, layer_out, activation=nn.ReLU()):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            layer_in,
            activation
        )
        self.decoder = nn.Sequential(
            layer_out,
            activation,
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded


class FiveLayerNet(nn.Module):
    def __init__(self, layer1, layer2, layer3, layer4, layer5, activation=nn.ReLU()):
        super(FiveLayerNet, self).__init__()
        self.net = nn.Sequential(
            layer1,
            activation,
            layer2,
            activation,
            layer3,
            activation,
            layer4,
            activation,
            layer5
        )

    def forward(self, x):
        return self.net(x)


class CustomDataset(Dataset):
    def __init__(self, annotations_file, transform=None, train=True):
        if train:
            self.data = pd.read_csv(annotations_file).iloc[0:35000]
        else:
            self.data = pd.read_csv(annotations_file).iloc[35000:]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        target = self.data["RMSD"].iloc[idx]
        labels = self.data.iloc[idx, 1:9].values
        labels = torch.tensor(labels, dtype=torch.float32)  # Преобразуем данные в тензоры
        target = torch.tensor(target, dtype=torch.float32)
        if self.transform:
            labels = self.transform(labels)

        return labels, target


def train(model, device, train_loader, learning_rate=0.01, epochs=5, model_save_path='best_model.pth'):
    loss_fn = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.99), eps=1e-08, weight_decay=1e-5)
    history = []
    best_loss = float('inf')

    for epoch in tqdm(range(epochs)):
        epoch_loss = 0.0
        for batch, labels in train_loader:
            optimizer.zero_grad()

            batch = batch.float()
            batch, labels = batch.to(device), labels.to(device)

            outputs = model(batch)
            labels = labels.unsqueeze(1)

            loss = loss_fn(outputs, labels)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()


        average_loss = epoch_loss / len(train_loader)
        history.append(average_loss)

        if average_loss < best_loss:
            best_loss = average_loss
            torch.save(model.state_dict(), model_save_path)
            print(f'Model saved with loss {best_loss} at epoch {epoch + 1}')

        print(f'Epoch {epoch + 1}, Loss: {average_loss}')

    plt.plot(range(0, epochs), history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')
    plt.show()


def test(model, device, test_loader):
    model.eval()
    total_loss = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch, labels in test_loader:
            batch, labels = batch.to(device), labels.to(device)
            outputs = model(batch)

            predictions = outputs.squeeze()
            loss = mape(predictions, labels)
            total_loss += loss.item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    avg_loss = total_loss / len(test_loader.dataset)
    print(f"MAPE: {avg_loss:.4f}")

def mape(predictions, targets):
    epsilon = 1e-8
    targets_safe = targets + epsilon
    return torch.mean(torch.abs((targets_safe - predictions) / targets_safe)) * 100

def create_model_from_layers(layer_list):
    layers = []
    for layer in layer_list:
        if isinstance(layer, tuple):
            layer_type, *params = layer
            if layer_type == 'Linear':
                layers.append(nn.Linear(*params))
            elif layer_type == 'ReLU':
                layers.append(nn.ReLU())
            elif layer_type == 'Sigmoid':
                layers.append(nn.Sigmoid())
            else:
                raise ValueError(f"Неизвестный тип слоя: {layer_type}")
        elif isinstance(layer, nn.Module):
            layers.append(layer)
        else:
            raise TypeError("Элемент списка должен быть слоем nn.Module или tuple с параметрами.")

    return nn.Sequential(*layers)

def pretrain_layer(encoder_layer, next_layer, previous_layer, data_loader, activation=nn.ReLU(), epochs=2, lr=0.001):
    autoencoder = Autoencoder(encoder_layer, next_layer, activation)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=lr)
    if len(previous_layer) != 0:
        previous_layer_model = create_model_from_layers(previous_layer)

    for epoch in tqdm(range(epochs)):
        for batch, _ in data_loader:
            optimizer.zero_grad()
            batch = batch.float()
            if len(previous_layer) != 0:
                batch = previous_layer_model.forward(batch)
            decoded, encoded = autoencoder(batch)
            loss = criterion(decoded, batch)
            loss.backward()
            optimizer.step()
    return autoencoder.encoder[0]


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 128
    train_dataset = CustomDataset("CASP.csv", train=True)
    test_dataset = CustomDataset("CASP.csv", train=False)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    layers = []
    epochs = 20
    layer1 = nn.Linear(8, 6)
    layer1_2 = nn.Linear(6, 8)
    layer2 = nn.Linear(6, 4)
    layer2_2 = nn.Linear(4, 6)
    layer3 = nn.Linear(4, 3)
    layer3_1 = nn.Linear(3, 4)
    layer4 = nn.Linear(3, 2)
    layer4_1 = nn.Linear(2, 3)
    layer5 = nn.Linear(2, 1)

    layer1 = pretrain_layer(layer1,layer1_2, previous_layer=layers, data_loader=train_loader)
    layers.append(layer1)
    layer2 = pretrain_layer(layer2, layer2_2, previous_layer=layers, data_loader=train_loader)
    layers.append(layer2)
    layer3 = pretrain_layer(layer3, layer3_1, previous_layer=layers, data_loader=train_loader)
    layers.append(layer3)
    layer4 = pretrain_layer(layer4, layer4_1, previous_layer=layers, data_loader=train_loader)
    layers.append(layer4)

    model = FiveLayerNet(layer1, layer2, layer3, layer4, layer5)
    print(model)

    train(model,device, train_loader, epochs)
    test(model, device, train_loader)


if __name__ == "__main__":
    main()
