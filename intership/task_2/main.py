import torch
import torch.nn as nn
from torchvision.transforms import v2
from torch.utils.data import Dataset
from torchvision.io import read_image

import seaborn as sns

from tqdm import tqdm

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import json
import os


class ImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_index = self.read_json(annotations_file)
        self.img_labels = self.read_json("images.json")
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_index)

    def __getitem__(self, idx):
        image_id = self.img_index[idx]
        metadata_item = next(item for item in self.img_labels if item['id'] == image_id)
        img_path = os.path.join(self.img_dir, metadata_item["name"])
        image = read_image(img_path)
        image = image.float() / 255.0
        label = metadata_item["class"]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    @staticmethod
    def read_json(path):
        with open(path, "r") as file:
            data = json.load(file)
        return data


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 20, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(50 * 61 * 61, 500),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(500, 10)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x


def train(model, device, train_loader, learning_rate=0.0001, epochs=5, model_save_path='best_model.pth', fine_tuning=False):
    loss_fn = nn.CrossEntropyLoss().to(device)
    if fine_tuning:
        for param in model.classifier.parameters():
            param.requires_grad = True
        trainable_params = [p for p in model.parameters() if p.requires_grad is True]
        optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.99), eps=1e-08, weight_decay=1e-5)
    history = []
    best_loss = float('inf')

    for epoch in tqdm(range(epochs)):
        epoch_loss = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            pred = model(x)

            loss = loss_fn(pred, y)
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
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = correct / total
    print(f"Accuracy on the test set: {accuracy:.2%}")

    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


def main():
    batch_size_train = 16
    batch_size_test = 16
    transform = v2.Compose([
        v2.ToTensor(),
        v2.Resize((256, 256)),
        v2.Normalize((0.5,), (0.5,))
    ])
    train_dataset = ImageDataset("train.json", "data", transform=transform)
    test_dataset = ImageDataset("validation.json", "data", transform=transform)
    print(train_dataset[0])
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size_train, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size_test, shuffle=False
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)
    model.load_state_dict(torch.load("best_model.pth"))
    train(model, device, train_loader,learning_rate=0.01 ,epochs=10)
    test(model, device, test_loader)

    transform = v2.Compose([
        v2.Resize(256),
        v2.CenterCrop(224),
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_dataset = ImageDataset("train.json", "data", transform=transform)
    test_dataset = ImageDataset("validation.json", "data", transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size_train, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size_test, shuffle=False
    )

    model_squeezenet = torch.hub.load('pytorch/vision:v0.10.0', 'squeezenet1_1', weights="IMAGENET1K_V1")
    model_squeezenet.classifier[1].out_channels = 5
    model_squeezenet.classifier[0] = nn.BatchNorm2d(512)
    model_squeezenet = model_squeezenet.to(device)
    for param in model_squeezenet.parameters():
        param.requires_grad = False
    train(model_squeezenet, device, train_loader, learning_rate=0.001 ,epochs=20, fine_tuning=True)
    test(model_squeezenet, device, test_loader)


if __name__ == "__main__":
    main()
