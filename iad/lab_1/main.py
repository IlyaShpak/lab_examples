from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def preprocess_data(path: str, test_size=0.2, random_state=42):
    data = pd.read_csv(path)
    return train_test_split(data, test_size=0.2, random_state=42)

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
    data = np.array(pd.read_csv("seeds.csv"))
    pca_data, pca_error = pca(data, n_components)
    print("pca_error", pca_error)
    plt.figure(figsize=(8, 6))
    sns.barplot(x=np.arange(1, n_components+1), y=np.var(pca_data, axis=0) / np.sum(np.var(pca_data, axis=0)))
    plt.title('Важность компонент')
    plt.xlabel('Компонента')
    plt.ylabel('Доля объясненной дисперсии')
    plt.show()


def main():
    train_pca(2)
    train_pca(3)


if __name__ == "__main__":
    main()
