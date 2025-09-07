import os
import re
import tifffile
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm
# Crop Cells IF images into patch
root_path = "/root/autodl-tmp/Code_files/2025.2.4_YOLOv11-seg/data/source_data/Cells"

data_folder = os.path.join(root_path, 'PNGTIFF')
mask_folder = os.path.join(root_path, 'Masks')
pre_process_path = os.path.join(root_path, 'Cells_preprocess')

os.makedirs(pre_process_path, exist_ok=True)

def extract_patches(image_path, seg_path, patch_size, save_patch_dir, save_seg_dir, step=200):
    """
    Create Patches From TIFF
    """
    original_image = tifffile.imread(image_path)
    seg_image = np.array(Image.open(seg_path))
    
    if len(original_image.shape) == 3:
        channels, height, width = original_image.shape
    else:
        height, width = original_image.shape
        channels = 1
        original_image = original_image[np.newaxis, ...]
    
    saved_patches = []
    os.makedirs(save_patch_dir, exist_ok=True)
    os.makedirs(save_seg_dir, exist_ok=True)
    
    for y in range(0, height - patch_size[0] + 1, step):
        for x in range(0, width - patch_size[1] + 1, step):
            patch = original_image[:, y:y+patch_size[0], x:x+patch_size[1]]
            seg_patch = seg_image[y:y+patch_size[0], x:x+patch_size[1]]
            
            if np.sum(seg_patch) != 0:
                saved_patches.append(((y, x), patch))
                patch_filename = os.path.join(save_patch_dir, f"patch_x_{x}_{x + patch_size[1]}_y_{y}_{y + patch_size[0]}.tif")
                seg_filename = os.path.join(save_seg_dir, f"seg_x_{x}_{x + patch_size[1]}_y_{y}_{y + patch_size[0]}.png")
                
                tifffile.imwrite(patch_filename, patch)
                Image.fromarray(seg_patch).save(seg_filename)
    
    return saved_patches


def main():
    step = 200
    patch_size = (512, 512)
    total_patches = 0
    total_data = []
    
    for dirs in os.listdir(data_folder):
        if "Mask_NO" in dirs:
            number_match = re.search(r'\d+', dirs)
            if number_match:
                number = number_match.group(0)
                subdir_path = os.path.join(data_folder, dirs)
                file_path = os.path.join(subdir_path, f'{number}.tif')
                mask_path = os.path.join(mask_folder, f'MASK_NO{number}.png')
                if os.path.exists(file_path) and os.path.exists(mask_path):
                    total_data.append((number, dirs, file_path, mask_path))
                else:
                    print(f"{file_path} {mask_path} Not Valid.")
    
    for number, dirs, file_path, mask_path in tqdm(total_data, desc="Processing patches"):
        save_patch_dir = os.path.join(pre_process_path, str(number), 'patch')
        save_seg_dir = os.path.join(pre_process_path, str(number), 'seg')
        os.makedirs(save_patch_dir, exist_ok=True)
        os.makedirs(save_seg_dir, exist_ok=True)
        
        patches = extract_patches(file_path, mask_path, patch_size, save_patch_dir, save_seg_dir, step)
        total_patches += len(patches)
    
    print(f"Number : {total_patches}")


if __name__ == "__main__":
    main()
