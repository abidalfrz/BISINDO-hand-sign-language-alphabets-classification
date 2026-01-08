import os
import numpy as np
import cv2
import torch
from torchvision.transforms import v2

def read_img(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.uint8)
    return img

def preprocess_images(img):
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img_transform = transform(img)
    return img_transform.unsqueeze(0)  # Add batch dimension