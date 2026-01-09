import os
import numpy as np
import cv2
import torch
from torchvision.transforms import v2
import mediapipe as mp

mp_holistic = mp.solutions.holistic

def get_union_bbox(results, h, w, padding=30):
    x_coords = []
    y_coords = []

    if results.left_hand_landmarks:
        x_coords.extend([lm.x * w for lm in results.left_hand_landmarks.landmark])
        y_coords.extend([lm.y * h for lm in results.left_hand_landmarks.landmark])
    
    if results.right_hand_landmarks:
        x_coords.extend([lm.x * w for lm in results.right_hand_landmarks.landmark])
        y_coords.extend([lm.y * h for lm in results.right_hand_landmarks.landmark])

    if not x_coords:
        return None

    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    box_w = max_x - min_x
    box_h = max_y - min_y

    box_size = max(box_w, box_h) + (padding * 2)
    
    center_x = min_x + box_w / 2
    center_y = min_y + box_h / 2

    x1 = int(center_x - box_size / 2)
    y1 = int(center_y - box_size / 2)
    x2 = int(center_x + box_size / 2)
    y2 = int(center_y + box_size / 2)

    return x1, y1, x2, y2

def crop_with_pad(img, bbox, target_size=(224, 224)):
    h, w, _ = img.shape
    x1, y1, x2, y2 = bbox
    
    box_w = x2 - x1
    box_h = y2 - y1
    
    canvas = np.zeros((box_h, box_w, 3), dtype=np.uint8)
    
    src_x1 = max(0, x1)
    src_y1 = max(0, y1)
    src_x2 = min(w, x2)
    src_y2 = min(h, y2)
    
    dst_x1 = src_x1 - x1
    dst_y1 = src_y1 - y1
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)
    
    if src_x2 > src_x1 and src_y2 > src_y1:
        canvas[dst_y1:dst_y2, dst_x1:dst_x2] = img[src_y1:src_y2, src_x1:src_x2]
    
    return cv2.resize(canvas, target_size)


def process_and_crop_dataset(img, target_size=(224, 224)):
    
    with mp_holistic.Holistic(static_image_mode=True, min_detection_confidence=0.5) as holistic:
                
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        
        results = holistic.process(img_rgb)
        
        bbox = get_union_bbox(results, h, w, padding=30)
        
        if bbox:
            final_img = crop_with_pad(img_rgb, bbox, target_size=target_size)
        else:
            # Fallback
            final_img = cv2.resize(img_rgb, target_size)

    return final_img

def preprocess_images(img, target_size=(224, 224)):
    img = process_and_crop_dataset(img, target_size=target_size)
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img_transform = transform(img)
    return img_transform.unsqueeze(0)  # Add batch dimension