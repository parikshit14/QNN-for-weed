import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import os
import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from PIL import Image
from sklearn.model_selection import train_test_split


csv_file_path = "prepare_data/deepweeds/labels/labels.csv"
df = pd.read_csv(csv_file_path)
string_to_append = "prepare_data/deepweeds/images/"
df["Filename"] = df["Filename"].apply(lambda x: string_to_append + x)
classes = df.set_index("Label")["Species"].to_dict()
num_classes = len(classes)


class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[
            idx, 0
        ]  # Assuming the image path is in the first column
        image = Image.open(img_name)
        label = int(
            self.dataframe.iloc[idx, 1]
        )  # Assuming the label is in the second column

        if self.transform:
            image = self.transform(image)

        return image, label


train_df, temp_df = train_test_split(df, test_size=0.4, random_state=42)

# Split the temporary set into validation and test sets
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
data_transforms_train = transforms.Compose(
    [
        transforms.Resize(256),  # Resize to 256x256
        transforms.RandomRotation(360),  # Random rotation in the range [-360, +360]
        transforms.RandomResizedCrop(224),  # Random crop to 224x224
        transforms.ColorJitter(
            brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
        ),  # Random color jitter
        transforms.RandomPerspective(
            distortion_scale=0.5
        ),  # Random perspective transformation
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

data_transforms_val_test = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def get_dataloaders():
    train_dataset = CustomDataset(dataframe=train_df, transform=data_transforms_train)
    val_dataset = CustomDataset(dataframe=val_df, transform=data_transforms_val_test)
    test_dataset = CustomDataset(dataframe=test_df, transform=data_transforms_val_test)

    # Create data loaders for training, validation, and testing
    batch_size = 32
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
