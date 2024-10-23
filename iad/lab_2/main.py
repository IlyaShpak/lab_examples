import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE

from matplotlib import pyplot as plt

import numpy as np
import pandas as pd


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            torch.nn.Linear(input_dim, 4),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(4, encoding_dim),
            torch.nn.LeakyReLU()
        )
        self.decoder = nn.Sequential(
            torch.nn.Linear(encoding_dim, 4),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(4, input_dim),
            torch.nn.LeakyReLU()
        )

    def forward(self, x):
        code = self.encoder(x)
        reconstructed = self.decoder(code)
        return reconstructed

    def get_code(self, x):
        code = self.encoder(x)
        return code


def plot_latent_space_3d(ax, model, X_test, y_test):
    with torch.no_grad():
        codes = model.get_code(X_test).numpy()

    for label in np.unique(y_test):
        idx = y_test == label
        ax.scatter(codes[idx, 0], codes[idx, 1], codes[idx, 2], label=f"Class {label}")

    ax.set_title("Latent space (3D)")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_zlabel("Component 3")
    ax.legend()


def plot_latent_space(ax, model, X_test, y_test):
    with torch.no_grad():
        codes = model.get_code(X_test).numpy()

    for label in np.unique(y_test):
        idx = y_test == label
        ax.scatter(codes[idx, 0], codes[idx, 1], label=f"Class {label}")

    ax.set_title("Latent space (2D)")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend()

def plot_tsne(model, X_test, y_test, n_components=2):
    with torch.no_grad():
        codes = model.get_code(X_test).numpy()

    tsne = TSNE(n_components=n_components)
    codes_tsne = tsne.fit_transform(codes)

    if n_components == 2:
        plt.figure(figsize=(8, 6))
        for label in np.unique(y_test):
            idx = y_test == label
            plt.scatter(codes_tsne[idx, 0], codes_tsne[idx, 1], label=f"Class {label}")
        plt.title("t-SNE (2D projection)")
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")
        plt.legend()
        plt.show()
    elif n_components == 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        for label in np.unique(y_test):
            idx = y_test == label
            ax.scatter(codes_tsne[idx, 0], codes_tsne[idx, 1], codes_tsne[idx, 2], label=f"Class {label}")
        ax.set_title("t-SNE (3D projection)")
        ax.set_xlabel("t-SNE Component 1")
        ax.set_ylabel("t-SNE Component 2")
        ax.set_zlabel("t-SNE Component 3")
        ax.legend()
        plt.show()

def train_autoencoder(encoding_dim=2):
    data_set = pd.read_csv("seeds.csv")
    y = data_set['V8']
    X = data_set.drop(columns=['V8'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    X_train = torch.FloatTensor(np.array(X_train))
    X_test = torch.FloatTensor(np.array(X_test))
    y_test = np.array(y_test)  # Для графика

    train_loader = DataLoader(TensorDataset(X_train, X_train), batch_size=4, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, X_test), batch_size=4)

    input_dim = X_train.shape[1]
    model = AutoEncoder(input_dim, encoding_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05, betas=(0.9, 0.999))

    epochs = 50
    history = []

    for epoch in range(epochs):
        running_loss = 0.0
        for data in train_loader:
            inputs, _ = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        history.append(running_loss / len(train_loader))
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader)}")

    with torch.no_grad():
        test_loss = 0.0
        for data in test_loader:
            inputs, _ = data
            outputs = model(inputs)
            test_loss += criterion(outputs, inputs).item()
        print(f"Test Loss: {test_loss / len(test_loader)}")

    return model, history, X_test, y_test


def main():
    model_2d, history_2d, X_test_2d, y_test_2d = train_autoencoder(2)
    model_3d, history_3d, X_test_3d, y_test_3d = train_autoencoder(3)

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    axs[0, 0].plot(history_2d)
    axs[0, 0].set_title("Training Loss (2D)")
    axs[0, 0].set_xlabel("Epoch")
    axs[0, 0].set_ylabel("Loss")

    plot_latent_space(axs[0, 1], model_2d, X_test_2d, y_test_2d)

    axs[1, 0].plot(history_3d)
    axs[1, 0].set_title("Training Loss (3D)")
    axs[1, 0].set_xlabel("Epoch")
    axs[1, 0].set_ylabel("Loss")

    ax_3d = fig.add_subplot(224, projection='3d')
    plot_latent_space_3d(ax_3d, model_3d, X_test_3d, y_test_3d)

    plt.tight_layout()
    plt.show()

    plot_tsne(model_2d, X_test_2d, y_test_2d, n_components=2)
    plot_tsne(model_3d, X_test_3d, y_test_3d, n_components=3)


if __name__ == "__main__":
    main()
