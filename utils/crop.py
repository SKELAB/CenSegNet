from PIL import Image
import math
import numpy as np

import torch
import torch.nn.functional as F


def pad_image_right_bottom(image, patch_size):
    width, height = image.size

    new_width = math.ceil(width / patch_size) * patch_size
    new_height = math.ceil(height / patch_size) * patch_size

    right_padding = new_width - width
    bottom_padding = new_height - height

    padded_image = Image.new("RGB", (new_width, new_height), color=(0, 0, 0))  
    padded_image.paste(image, (0, 0))  

    return padded_image, right_padding, bottom_padding


def slice_image_with_padding(image, patch_size=500, overlap=0.2, padding=50, left_padding=0,pad_mod="reflect"):
    width, height = image.size
    stride = int(patch_size * (1 - overlap))  

    patches = []
    patches_unpad = []
    positions = []  

    y_positions = list(range(0, height - patch_size + 1, stride))
    x_positions = list(range(0, width - patch_size + 1, stride))

    for y in y_positions:
        for x in x_positions:
            patch = image.crop((x, y, x + patch_size, y + patch_size))

            patch_np = np.array(patch)
            if pad_mod=="reflect":
                padded_patch_np = np.pad(
                    patch_np, 
                    ((padding, padding), (padding, padding), (0, 0)),  # 3D padding (H, W, C)
                    mode='reflect'  
                )
            else:
                padded_patch_np = np.pad(
                    patch_np, 
                    ((padding, padding), (padding, padding), (0, 0)),  # 3D padding (H, W, C)
                    mode='constant',  # Constant padding mode
                    constant_values=0  
                )

            padded_patch = Image.fromarray(padded_patch_np)
            un_padded_patch = Image.fromarray(patch_np)

            positions.append((x + left_padding, y))

            patches.append(padded_patch)
            patches_unpad.append(un_padded_patch)

    return patches, patches_unpad, positions



def resize_and_unpad(tensor,padding,size=(256, 256)):
    """
    (256,256) -> (250,250)
    """
    resized = F.interpolate(tensor, size=size, mode='bilinear', align_corners=False)

    unpadded = resized[:, :, padding:-padding, padding:-padding]

    return unpadded