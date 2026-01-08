import os
import torch
import numpy as np
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
from torch.nn import Module
import pickle

class SignLanguageModel(Module):
    def __init__(self, hidden_size, dropout_size, num_classes, freeze_backbone=False):
        super(SignLanguageModel, self).__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.backbone.fc = nn.Sequential(
            nn.Linear(self.backbone.fc.in_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_size),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../artifacts/best_model.pth")
ENCODER_PATH = os.path.join(os.path.dirname(__file__), "../artifacts/encoder.pkl")

class Predictor:
    def __init__(self, model_path=MODEL_PATH, encoder_path=ENCODER_PATH, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with open(encoder_path, 'rb') as f:
            self.encoder = pickle.load(f)
        
        num_classes = len(self.encoder.classes_)

        self.model = SignLanguageModel(hidden_size=128, dropout_size=0.2, num_classes=num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()


    def predict(self, img_transform):
        img_transform = img_transform.to(self.device)
        with torch.no_grad():
            outputs = self.model(img_transform)
            _, predicted = torch.max(outputs, 1)
            conf = nn.Softmax(dim=1)(outputs)
            predicted_label = self.encoder.inverse_transform(predicted.cpu().numpy())
        return predicted_label[0], conf[0][predicted].item()
    
predictor = Predictor()



