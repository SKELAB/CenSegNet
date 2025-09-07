import os
import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from tqdm import tqdm
from torch.utils.data import Dataset

class CoCoSegmentationDataset(Dataset):
    def __init__(self, coco_json, img_dir, transform=None, target_transform=None, pad_size=15):
        """
        Initialize dataset
        :param coco_json: annotation file path (COCO format as JSON, YOLO format as TXT)
        :param img_dir: image folder path
        :param data_format: data format, supports 'coco' or 'yolo'
        :param transform: image transform
        :param target_transform: mask transform
        :param pad_size: bounding box padding size
        """
        self.data_path = coco_json
        self.img_dir = img_dir

        self.transform = transform
        self.target_transform = target_transform
        self.pad_size = pad_size
        self.total_instances = []

        # Load data based on format
        self._load_coco_data()

    def _load_coco_data(self):
        """Load COCO format data"""
        self.coco = COCO(self.data_path)
        self.image_ids = self.coco.getImgIds()

        for image_id in tqdm(self.image_ids):
            image_info = self.coco.loadImgs(image_id)[0]
            image_path = os.path.join(self.img_dir, image_info['file_name'])
            image = cv2.imread(image_path)
            ann_ids = self.coco.getAnnIds(imgIds=image_id)
            annotations = self.coco.loadAnns(ann_ids)
            # print(annotations)
            for ann in annotations:
                bbox = ann['bbox']  # [x, y, width, height]
                segm = ann['segmentation']
                if isinstance(segm, list):  # Polygon format
                    try:
                        rle = self.coco.annToRLE(ann)
                        instance_mask = maskUtils.decode(rle)
                    except TypeError:
                        # If conversion fails, try creating mask directly from polygon
                        h, w = image.shape[:2]
                        instance_mask = np.zeros((h, w), dtype=np.uint8)
                        # Convert polygon to mask
                        polygon = np.array(segm[0]).reshape(-1, 2)
                        cv2.fillPoly(instance_mask, [polygon.astype(np.int32)], 1)
                        print(f"annotation ID {ann['id']} RLE transfer failed, use poly filling method...")
                        
                elif isinstance(segm, dict):  # RLE format
                    instance_mask = maskUtils.decode(segm)
                else:
                    print(f"Warning: Unsupported segmentation format for annotation {ann['id']}")
                    continue

                instance_info = {
                    'image_path': image_path,
                    'image_shape': image.shape[:2],
                    'bbox': bbox,
                    'mask': instance_mask
                }
                self.total_instances.append(instance_info)

    def __len__(self):
        return len(self.total_instances)

    def __getitem__(self, idx):
        instance = self.total_instances[idx]
        image = cv2.imread(instance['image_path'])
        image_shape = instance['image_shape']
        bbox = instance['bbox']
        instance_mask = instance['mask']
        instance_mask[instance_mask > 0] = 255
        
        x_min, y_min, w, h = map(int, bbox)
        x_max, y_max = x_min + w, y_min + h

        x_min = max(x_min - self.pad_size, 0)
        y_min = max(y_min - self.pad_size, 0)
        x_max = min(x_max + self.pad_size, image_shape[1])
        y_max = min(y_max + self.pad_size, image_shape[0])

        cropped_image_raw = image[y_min:y_max, x_min:x_max]
        cropped_image_norm = (cropped_image_raw - cropped_image_raw.min()) / (cropped_image_raw.max() - cropped_image_raw.min()) * 255  # Min-Max normalization
        cropped_image = cropped_image_norm.astype(np.uint8)
        cropped_mask = instance_mask[y_min:y_max, x_min:x_max]

        image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)

        

        if self.transform:
            image_rgb = self.transform(image_rgb)
        if self.target_transform:
            cropped_mask = self.target_transform(cropped_mask)

        return image_rgb, cropped_mask

