import os
import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from tqdm import tqdm
from torch.utils.data import Dataset

def pad_to_square_with_extra_pad(x_min, x_max, y_min, y_max, image_shape):
    """
    First adjust the bounding box to a square, then add an extra padsize,
    while ensuring it does not exceed the image boundaries.

    :param x_min: original x_min
    :param x_max: original x_max
    :param y_min: original y_min
    :param y_max: original y_max
    :param image_shape: (image height, image width)
    :param padsize: extra padding size
    :return: (x_min_padded, x_max_padded, y_min_padded, y_max_padded)
    """
    img_h, img_w = image_shape
    
    # Calculate current width and height
    width = x_max - x_min
    height = y_max - y_min

    # To make it square, using the longer side as reference
    max_side = max(width, height)

    # Calculate padding to make it max_side x max_side
    pad_w = (max_side - width) / 2
    pad_h = (max_side - height) / 2

    pad_left = int(pad_w)
    pad_right = int(pad_w + 0.5)  # Avoid floating-point issues
    pad_top = int(pad_h)
    pad_bottom = int(pad_h + 0.5)

    # Apply padding and ensure it does not exceed boundaries
    x_min_padded = max(x_min - pad_left, 0)
    if x_min_padded == 0:  # If left side exceeds image boundary
        x_max_padded = min(x_max + pad_right + (pad_left-x_min), img_w)  # Add the exceeded part to the right side
    else:
        x_max_padded = min(x_max + pad_right, img_w)  # Normally only adjust right padding
    # Adjust left padding if right side exceeds boundary
    if x_max_padded >= img_w:
        x_min_padded = max(x_min_padded - (x_max + pad_right - img_w), 0)


    y_min_padded = max(y_min - pad_top, 0)
    if y_min_padded == 0:  # If top side exceeds image boundary
        y_max_padded = min(y_max + pad_bottom + (pad_top-y_min), img_h)  # Add the exceeded part to the bottom
    else:
        y_max_padded = min(y_max + pad_bottom, img_h)  # Normally only adjust bottom padding
    if y_max_padded >= img_h:
        y_min_padded = min(y_min_padded - (y_max + pad_bottom - img_h), img_h)

    return x_min_padded, x_max_padded, y_min_padded, y_max_padded


def pad_bbox_with_adjustment(x_min, x_max, y_min, y_max, image_shape, padsize):
    # Calculate initial padding
    img_h, img_w = image_shape
    x_min_padded = x_min - padsize
    x_max_padded = x_max + padsize
    row=0
    if x_min_padded <0:
        x_max_padded += abs(x_min - padsize)
        x_min_padded=0
        row+=1
    if x_max_padded >img_w:
        x_min_padded-=abs(x_max_padded - img_w)
        x_max_padded=img_w
        row+=1
    if row ==2:
        x_min_padded = 0
        x_max_padded = img_w

    column=0
    y_min_padded=y_min - padsize
    y_max_padded=y_max + padsize
    if y_min_padded<0:
        y_max_padded += abs(y_min - padsize)
        y_min_padded=0
        column+=1
    if y_max_padded>img_h:
        y_min_padded-=abs(y_max_padded - img_h)
        y_max_padded=img_h
        column+=1
    if column==2:
        y_min_padded = 0
        y_max_padded = img_h

    return x_min_padded, x_max_padded, y_min_padded, y_max_padded


class EpithelialInstanceSegmentationDataset(Dataset):
    def __init__(self, coco_json, img_dir, transform=None, target_transform=None, pad_size=15,thresold=400):
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
        self.thresold = thresold
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
        
        # Crop image and mask
        x_min_o, y_min_o, w, h = map(int, bbox)
        x_max_o, y_max_o= x_min_o + w, y_min_o + h

        x_min_padded, x_max_padded, y_min_padded, y_max_padded = pad_to_square_with_extra_pad(x_min_o, x_max_o, y_min_o, y_max_o, image_shape)
        
        if x_max_padded -  x_min_padded <=self.thresold:
            x_min_padded, x_max_padded, y_min_padded, y_max_padded= pad_bbox_with_adjustment(x_min_padded, x_max_padded, y_min_padded, y_max_padded, image_shape, padsize=self.pad_size)
        

        cropped_image_raw = image[y_min_padded:y_max_padded, x_min_padded:x_max_padded]
        cropped_image_norm = (cropped_image_raw - cropped_image_raw.min()) / (cropped_image_raw.max() - cropped_image_raw.min()) * 255  # Min-Max normalization
        cropped_image = cropped_image_norm.astype(np.uint8)
        image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)


        cropped_mask = instance_mask[y_min_padded:y_max_padded, x_min_padded:x_max_padded]

        # Apply transforms
        if self.transform:
            image_rgb = self.transform(image_rgb)
        if self.target_transform:
            cropped_mask = self.target_transform(cropped_mask)

        return image_rgb, cropped_mask


