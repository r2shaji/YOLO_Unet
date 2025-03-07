import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import albumentations as albu
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


def xywhn_to_xyxy(detection, height, width):
    cx, cy, w, h = detection
        
    x1 = int((cx - w / 2) * width)
    y1 = int((cy - h / 2) * height)
    x2 = int((cx + w / 2) * width)
    y2 = int((cy + h / 2) * height)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)

    return x1, y1, x2, y2

def load_label(image_path,label_folder):
        
    label_file_name = os.path.basename(image_path).replace('.jpg', '.txt')
    label_file_path = os.path.join(label_folder,label_file_name)
    with open(label_file_path, 'r') as f:
        lines = f.readlines()
    lines= [line.rstrip() for line in lines]
    lines= [line.split() for line in lines]
    lines = np.array(lines).astype(float)
    lines = torch.from_numpy(lines)
    lines = sorted(lines, key=lambda x: x[1])
    sorted_labels = torch.tensor([t[0] for t in lines],  dtype=torch.float)
    sorted_labels = sorted_labels.long()
    sorted_boxes = [t[1:] for t in lines]
    ground_truth = { "sorted_labels": sorted_labels, "sorted_boxes_xywhn":sorted_boxes}
    return ground_truth



def create_data_loader(X, y, batch_size=32):
    print("shape of X",X.shape)
    print("shape of y",y.shape)
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return data_loader


def transform_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = np.array(image)
    normalize = albu.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    image = normalize(image=image)["image"]
    image = transforms.ToTensor()(image).unsqueeze(0)
    return image


class ReconstructionDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
