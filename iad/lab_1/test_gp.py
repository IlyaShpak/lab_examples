from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def preprocess_data(path: str, test_size=0.2, random_state=42):
    data = pd.read_csv(path)
    X = data.iloc[:, :-1].values  # Все колонки, кроме последней (признаки)
    y = data.iloc[:, -1].values  # Последняя колонка (метки классов)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def pca(X, n_components):
    X_centered = X - np.mean(X, axis=0)
    covariance_matrix = np.cov(X_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    top_eigenvectors = eigenvectors[:, :n_components]
    X_pca = np.dot(X_centered, top_eigenvectors)
    X_reconstructed = np.dot(X_pca, top_eigenvectors.T)
    reconstruction_error = np.mean(np.square(X_centered - X_reconstructed))
    return X_pca, reconstruction_error


def train_pca(n_components=2):
    # Загружаем и делим данные
    X_train, X_test, y_train, y_test = preprocess_data("seeds.csv")

    # Применяем PCA к тренировочным данным
    pca_data, pca_error = pca(X_train, n_components)
    print("PCA reconstruction error:", pca_error)

    # Визуализируем данные после PCA
    plt.figure(figsize=(8, 6))

    for class_value in np.unique(y_train):
        plt.scatter(pca_data[y_train == class_value, 0], pca_data[y_train == class_value, 1],
                    label=f'Class {int(class_value)}')

    plt.title(f'PCA with {n_components} Components')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.show()

    # Дополнительно визуализируем важность компонент
    plt.figure(figsize=(8, 6))
    sns.barplot(x=np.arange(1, n_components + 1), y=np.var(pca_data, axis=0) / np.sum(np.var(pca_data, axis=0)))
    plt.title('Важность компонент')
    plt.xlabel('Компонента')
    plt.ylabel('Доля объясненной дисперсии')
    plt.show()


def main():
    train_pca(2)  # PCA с 2 компонентами для визуализации
    train_pca(3)  # Для анализа с 3 компонентами


if __name__ == "__main__":
    main()
