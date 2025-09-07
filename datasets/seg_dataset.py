import os
import json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image

class SegmentationDataset(Dataset):
    def __init__(self, json_path, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        
        # Load COCO annotations
        with open(json_path, 'r') as f:
            self.coco_data = json.load(f)
        
        self.images = self.coco_data['images']
        self.annotations = self.coco_data['annotations']
        self.categories = self.coco_data['categories']
        
        # Create a mapping from image_id to its annotations
        self.image_id_to_annotations = {}
        for ann in self.annotations:
            self.image_id_to_annotations.setdefault(ann['image_id'], []).append(ann)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_info = self.images[idx]
        image_path = os.path.join(self.image_dir, image_info['file_name'])
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Create an empty mask
        mask = np.zeros(np.array(image).shape[:2], dtype=np.uint8)
        
        # Get annotations for the current image
        annotations = self.image_id_to_annotations.get(image_info['id'], [])
        
        # Fill mask with segmentation data
        for annotation in annotations:
            for segment in annotation['segmentation']:
                segment = np.array(segment).reshape((-1, 2))
                cv2.fillPoly(mask, [segment.astype(np.int32)], color=(255))
        
        mask = Image.fromarray(mask)
        
        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        return image, mask

# Examples:
# dataset = COCOSegmentationDataset("./data/train_data/Cells_channel_1/train.json", "./data/train_data/Cells_channel_1/yolo/images/train")
# image, mask = dataset[0]
