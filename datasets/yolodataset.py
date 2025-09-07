import os
import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

class YOLOSegmentationDataset(Dataset):
    """
    A PyTorch Dataset for handling YOLO format segmentation data.
    Handles both bounding boxes and segmentation masks in YOLO format.
    """
    
    def __init__(
        self,
        txt_path: str,
        img_dir: str,
        img_size: Tuple[int, int] = (640, 640),
        transform: Optional[transforms.Compose] = None,
        target_transform: Optional[transforms.Compose] = None,
        pad_size: int = 15,
        cache_images: bool = False
    ):
        """
        Initialize the YOLO segmentation dataset.
        
        Args:
            txt_path (str): Path to YOLO format annotation txt file
            img_dir (str): Directory containing the images
            img_size (tuple): Target image size (height, width)
            transform: Optional transforms to be applied to images
            target_transform: Optional transforms to be applied to masks
            pad_size (int): Padding size around bounding boxes
            cache_images (bool): Whether to cache images in memory
        """
        self.txt_path = txt_path
        self.img_dir = img_dir
        self.img_size = img_size
        self.transform = transform
        self.target_transform = target_transform
        self.pad_size = pad_size
        self.cache_images = cache_images
        
        # Initialize cache
        self.image_cache = {} if cache_images else None
        
        # Load and process all instances
        self.total_instances = self._load_instances()
        
    def _load_instances(self) -> list:
        """
        Load and process all instances from the YOLO format dataset.
        Returns a list of instance information dictionaries.
        """
        instances = []
        
        # Read annotation file
        with open(self.txt_path, 'r') as f:
            lines = f.readlines()
            
        print("Loading YOLO instances...")
        for line in tqdm(lines):
            parts = line.strip().split()
            if len(parts) < 5:  # Must have at least class and bbox
                continue
                
            image_path = os.path.join(self.img_dir, parts[0])
            
            # Verify image exists
            if not os.path.exists(image_path):
                print(f"Warning: Image not found: {image_path}")
                continue
            
            # Get image dimensions
            if self.cache_images:
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Warning: Could not load image: {image_path}")
                    continue
                self.image_cache[image_path] = image
                image_shape = image.shape[:2]
            else:
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Warning: Could not load image: {image_path}")
                    continue
                image_shape = image.shape[:2]
            
            # Process instance
            instance_info = self._process_annotation(parts[1:], image_path, image_shape)
            if instance_info is not None:
                instances.append(instance_info)
                
        return instances
    
    def _process_annotation(
        self, 
        annotation: List[str], 
        image_path: str, 
        image_shape: Tuple[int, int]
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single YOLO format annotation into instance information.
        
        Args:
            annotation: List of annotation values
            image_path: Path to the image
            image_shape: Shape of the image (height, width)
            
        Returns:
            Dictionary containing instance information or None if invalid
        """
        try:
            # Parse class and bbox
            class_id = int(annotation[0])
            bbox_values = list(map(float, annotation[1:5]))
            
            # Convert normalized YOLO bbox to absolute coordinates
            x_center, y_center, width, height = bbox_values
            image_height, image_width = image_shape
            
            x_min = int((x_center - width/2) * image_width)
            y_min = int((y_center - height/2) * image_height)
            width = int(width * image_width)
            height = int(height * image_height)
            
            # Create bbox in [x_min, y_min, width, height] format
            bbox = [x_min, y_min, width, height]
            
            # Create binary mask from bbox
            mask = np.zeros(image_shape, dtype=np.uint8)
            mask[y_min:y_min+height, x_min:x_min+width] = 1
            
            # If segmentation points are provided (after bbox coordinates)
            if len(annotation) > 5:
                seg_points = list(map(float, annotation[5:]))
                if len(seg_points) % 2 == 0:  # Must have even number of coordinates
                    # Convert normalized coordinates to absolute
                    points = []
                    for i in range(0, len(seg_points), 2):
                        x = int(seg_points[i] * image_width)
                        y = int(seg_points[i + 1] * image_height)
                        points.append([x, y])
                    
                    # Create mask from polygon
                    points = np.array(points, dtype=np.int32)
                    mask = np.zeros(image_shape, dtype=np.uint8)
                    cv2.fillPoly(mask, [points], 1)
            
            return {
                'image_path': image_path,
                'image_shape': image_shape,
                'bbox': bbox,
                'mask': mask,
                'category_id': class_id
            }
            
        except Exception as e:
            print(f"Error processing annotation: {e}")
            return None
    
    def _load_and_crop_image(
        self, 
        image_path: str,
        bbox: list,
        image_shape: Tuple[int, int]
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Load and crop image based on bbox with padding.
        
        Returns:
            Tuple of (cropped_image, (x_min, y_min, x_max, y_max))
        """
        # Get image
        if self.cache_images:
            image = self.image_cache[image_path]
        else:
            image = cv2.imread(image_path)
            
        # Calculate crop coordinates with padding
        x_min, y_min, w, h = map(int, bbox)
        x_max, y_max = x_min + w, y_min + h
        
        x_min = max(x_min - self.pad_size, 0)
        y_min = max(y_min - self.pad_size, 0)
        x_max = min(x_max + self.pad_size, image_shape[1])
        y_max = min(y_max + self.pad_size, image_shape[0])
        
        return image[y_min:y_max, x_min:x_max], (x_min, y_min, x_max, y_max)
    
    def __len__(self) -> int:
        """Return the total number of instances in the dataset."""
        return len(self.total_instances)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a single instance by index.
        
        Returns:
            Tuple of (image, mask) after processing and transforms
        """
        instance = self.total_instances[idx]
        
        # Load and crop image
        cropped_image, crop_coords = self._load_and_crop_image(
            instance['image_path'],
            instance['bbox'],
            instance['image_shape']
        )
        
        # Crop mask to match image
        x_min, y_min, x_max, y_max = crop_coords
        cropped_mask = instance['mask'][y_min:y_max, x_min:x_max]
        
        # Convert to RGB
        image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            image_rgb = self.transform(image_rgb)
        if self.target_transform:
            cropped_mask = self.target_transform(cropped_mask)
            
        return image_rgb, cropped_mask