import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Пример seed dataset (загрузим набор данных из CSV)
# Замените путь к файлу своим
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt"
columns = ['area', 'perimeter', 'compactness', 'length', 'width', 'asymmetry', 'groove', 'class']

# Загрузка данных
df = pd.read_csv(url, sep='\s+', header=None, names=columns)

# Отделяем признаки от меток классов
X = df.drop('class', axis=1)
y = df['class']

# Стандартизация признаков
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Инициализируем PCA, оставим 2 компоненты
pca = PCA(n_components=2)

# Применяем PCA к данным
X_pca = pca.fit_transform(X_scaled)

# Создаем DataFrame для визуализации
df_pca = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
df_pca['class'] = y

print(df_pca.head())

# Дополнительно можно отобразить результат на графике, если хотите
import matplotlib.pyplot as plt

# Визуализируем результаты PCA
plt.figure(figsize=(8, 6))
colors = ['r', 'g', 'b']
for i, color in zip([1, 2, 3], colors):
    plt.scatter(df_pca[df_pca['class'] == i]['PCA1'], df_pca[df_pca['class'] == i]['PCA2'],
                color=color, label=f'Class {i}')

plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.title('PCA on Seed Dataset')
plt.show()
