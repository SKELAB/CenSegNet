import os
import shutil
import json
import numpy as np
import cv2
from PIL import Image
from skimage import measure
from torchvision.datasets import CocoDetection
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import ultralytics
from ultralytics.data.converter import convert_coco
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

# Create centrosome data
SOURCE_PATH = "../data/source_data/centrosome/"
OUTPUT_JSON_DIR = "../data/train_data/centrosome/"
YOLO_PATH = "../data/train_data/centrosome/yolo"
TRAIN_TEST_SPLIT = 0.1  # Dataset split ratio

OBJECT="Centrosome"

# Data loading and preprocessing
def load_train_list(source_path):
    train_list = []
    source_path = Path(source_path)

    # Iterate through each folder in source_path
    for file in os.listdir(source_path):
        file_path = source_path / file
        patch_dir = file_path / 'patch'

        if patch_dir.is_dir():
            for patch in os.listdir(patch_dir):
                if patch.endswith(".png"):
                    patch_path = patch_dir / patch  
                    seg_path = patch_path.as_posix().replace("/patch/patch_", "/seg/gt/seg_")  # Convert to POSIX format and replace
                    train_list.append((str(patch_path), seg_path))
    
    return train_list

# COCO format conversion function
class CustomCocoDetection(CocoDetection):
    @staticmethod
    def create_coco_annotations(image_path, mask_path, coco_format, image_id, annotation_id, file_name):
        image = Image.open(image_path)
        mask = np.array(Image.open(mask_path))
        mask = np.where(mask > 0, 1, 0)  # Set all nonzero values to 1, keep zero values as 0

        image_info = {
            "id": image_id,
            "file_name": file_name,  # Use unified image path here
            "width": image.size[0],
            "height": image.size[1]
        }

        labeled_mask, num_features = measure.label(mask, background=0, return_num=True)
        if num_features != 0:
            coco_format['images'].append(image_info)
            for i in range(1, num_features + 1):
                single_object_mask = (labeled_mask == i).astype(np.uint8)
                # Skip if mask is empty
                if not np.any(single_object_mask):
                    continue
                
                if len(single_object_mask.shape) == 3:  # Check if it's a 3-channel image
                    single_object_mask = cv2.cvtColor(single_object_mask, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
                else:
                    single_object_mask = single_object_mask  
                # Get contours for segmentation
                contours, _ = cv2.findContours(
                    single_object_mask, 
                    cv2.RETR_EXTERNAL, 
                    cv2.CHAIN_APPROX_NONE
                )
                
                # Convert contours to COCO format segmentation
                segmentation = [
                    np.flip(contour, axis=1).ravel().tolist() 
                    for contour in contours
                ]
                segmentation = [list(map(int, contour)) for contour in segmentation]
                
                # Handle line segment cases (contours with too few points)
                if any(len(contour) < 6 for contour in segmentation):  # Each point has x,y, so <6 means <3 points
                    # Use dilation to thicken line segments
                    kernel = np.ones((2, 2), np.uint8)
                    single_object_mask = cv2.dilate(single_object_mask, kernel, iterations=1)
                    
                    # Recalculate contours
                    contours, _ = cv2.findContours(
                        single_object_mask, 
                        cv2.RETR_EXTERNAL, 
                        cv2.CHAIN_APPROX_NONE
                    )
                    
                    # Regenerate segmentation
                    segmentation = [
                        np.flip(contour, axis=1).ravel().tolist() 
                        for contour in contours
                    ]
                    segmentation = [list(map(int, contour)) for contour in segmentation]
                
                # Get bounding box coordinates
                x, y, w, h = cv2.boundingRect(single_object_mask)
                
                # Calculate area
                area = int(np.sum(single_object_mask))
                
                # Create annotation dictionary
                annotation = {
                    "id": annotation_id ,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "area": area,
                    "iscrowd": 0,
                    "segmentation": segmentation
                }
                coco_format['annotations'].append(annotation)
                annotation_id += 1
            image_id += 1

        return coco_format, image_id, annotation_id

# Save and generate COCO JSON
def generate_coco_json(train_list, output_json_path, split):
    image_id = 1
    annotation_id = 1
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": OBJECT, "supercategory": "none"}]
    }

    save_dirs = []
    for index in tqdm(range(len(train_list))):
        image_path, mask_path = train_list[index]
        filename = f"{image_id:012}.png"
        destination_path = os.path.join(YOLO_PATH, 'images', split, filename)
        shutil.copy(image_path, destination_path)
        save_dirs.append((image_path, destination_path, mask_path))

        coco_format, image_id, annotation_id = CustomCocoDetection.create_coco_annotations(
            image_path, mask_path, coco_format, image_id, annotation_id, filename
        )

    # Save COCO format JSON file
    with open(output_json_path, 'w') as json_file:
        json.dump(coco_format, json_file, indent=4)
    return save_dirs



def random_visualize(image_folder,coco_data):
    # Randomly select an image from the 'images' list
    image_info = random.choice(coco_data['images'])
    image_path = image_info['file_name']
    
    # Get all annotations for the selected image
    annotations = [annotation for annotation in coco_data['annotations'] if annotation['image_id'] == image_info['id']]
    
    # Create a new coco_data dict for this image and its annotations
    image_coco_data = {
        "images": [image_info],
        "annotations": annotations,
        "categories": coco_data['categories']
    }
    
    # Visualize the selected image with its annotations
    visualize_coco_annotations(os.path.join(image_folder,image_path), image_coco_data)


def visualize_coco_annotations(image_path, coco_annotations):
    # Load the image
    image = np.array(Image.open(image_path))
    print(image_path)
    img_bbox = image.copy()
    img_segmentation = image.copy()
    # Prepare a copy of the image for visualization
    img_vis = image.copy()

    # Loop through the annotations to visualize bounding boxes and segmentation
    for annotation in coco_annotations['annotations']:
        # Get the bounding box coordinates [x_min, y_min, width, height]
        x, y, w, h = annotation['bbox']
        
        # Draw the bounding box on the image (Green rectangle)
        cv2.rectangle(img_bbox, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
        
        # Draw the segmentation (Yellow polygon)
        for segment in annotation['segmentation']:
            segment = np.array(segment).reshape((-1, 2))
            cv2.fillPoly(img_segmentation, [segment.astype(np.int32)], color=(0, 255, 255))
            # cv2.polylines(img_segmentation, [segment.astype(np.int32)], isClosed=True, color=(0, 255, 255), thickness=2)

    # Create subplots: one row, three columns
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Image with bounding boxes
    axes[1].imshow(img_bbox)
    axes[1].set_title('Image with Bounding Boxes')
    axes[1].axis('off')
    
    # Image with segmentations
    axes[2].imshow(img_segmentation)
    axes[2].set_title('Image with Segmentations')
    axes[2].axis('off')
    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()


# Main function
def main():
    os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)
    os.makedirs(os.path.join(YOLO_PATH, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(YOLO_PATH, 'images', 'test'), exist_ok=True)

    # Data loading and splitting
    file_lists = load_train_list(SOURCE_PATH)
    train_set, test_set = train_test_split(file_lists, test_size=TRAIN_TEST_SPLIT, random_state=42)
    print("Training set:", len(train_set))
    print("Testing set:", len(test_set))

    # Generate COCO format JSON files
    save_dirs_total = []
    for split, data_list in zip(['train', 'test'], [train_set, test_set]):
        output_json_path = os.path.join(OUTPUT_JSON_DIR, f'{split}.json')
        save_dirs = generate_coco_json(data_list, output_json_path, split)
        save_dirs_total.append(save_dirs)

    # Convert to YOLO format
    convert_coco(OUTPUT_JSON_DIR, use_segments=True, use_keypoints=False, cls91to80=False)

    # Move and copy files to YOLO path
    shutil.move('./coco_converted/labels', YOLO_PATH)
    shutil.rmtree('./coco_converted/')

if __name__ == '__main__':
    main()
