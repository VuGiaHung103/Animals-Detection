from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import ToTensor, Resize, Compose

import os
import pickle
import numpy as np
import cv2

class AnimalDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        if train:
            mode = "Train"
        else:
            mode = "Test"
        self.root = os.path.join(root, mode)
        if transform is None:
            self.transform = Compose([
                Resize((200, 200)),
                ToTensor()
            ])
        else:
            self.transform = transform
        self.categories = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'sheep', 'spider', 'squirrel']
        self.images_path =[]
        self.labels =[]
        for i, category in enumerate(self.categories):
            data_file_path = os.path.join(self.root, category)
            for file_name in os.listdir(data_file_path):
                file_path = os.path.join(data_file_path, file_name)
                self.images_path.append(file_path)
                self.labels.append(i)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        image_path = self.images_path[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label


if __name__ == '__main__':
    root = "D:\\University\\AI\\DeepLearning\\animals"
    dataset = AnimalDataset(root=root, train = True)
    training_dataloader = DataLoader(
        dataset=dataset,
        batch_size=19,
        num_workers=4,
        shuffle=True,
        drop_last=True
    )
    for images, labels in training_dataloader:
        print(images.shape)
        print(labels)
