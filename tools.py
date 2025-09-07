import os
import cv2
import tifffile
import yaml
import numpy as np
from PIL import Image
import torch
import numpy as np
from PIL import Image
from utils.cv import generate_unet_mask, apply_color_mask,generate_unet_mask_EL
from utils.crop import resize_and_unpad, slice_image_with_padding, pad_image_right_bottom
from segmentors import unet_model
from ultralytics import YOLO
import argparse
from matplotlib import pyplot as plt
from tqdm import tqdm
from src.cv import read_image_as_pil
import os
import glob
import re


def get_image(image_file,mode="IF"):
    if mode =="IF":
        original_image = tifffile.imread(image_file)
        normalized_image = cv2.normalize(original_image[-1], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        rgb_image = cv2.cvtColor(normalized_image, cv2.COLOR_GRAY2RGB)
        whole_image = Image.fromarray(rgb_image)
    elif mode =="IHC":
        whole_image = read_image_as_pil(image_file)
    return whole_image

def process_patch_masks(
    padded_image: np.ndarray,
    final_result: list,
    patch_size: int,
    original_height: int,
    original_width: int,
    overlap_threshold: float = 0.5
):
    all_mask_coords = []
    final_mask = np.zeros(padded_image.size[::-1], dtype=np.uint8)
    final_mask_label = np.zeros(padded_image.size[::-1], dtype=np.uint32)

    for patch, patches_unpad, positions, final_patch_mask in final_result:
        x, y = positions
        if x + patch_size > final_mask.shape[1] or y + patch_size > final_mask.shape[0]:
            print(f"Patch out of bounds: ({x}, {y}), skipping...")
            continue

        y_indices, x_indices = np.where(final_patch_mask == 255)
        coords = {(patch_x + x, patch_y + y) for patch_y, patch_x in zip(y_indices, x_indices)}
        
        if coords:
            all_mask_coords.append(coords)

    all_mask_coords.sort(key=len, reverse=True)
    
    final_masks = {}
    current_label = 1

    for coords in all_mask_coords:
        coords_set = frozenset(coords)
        should_add = True
        
        for existing_coords in final_masks:
            overlap = len(coords & existing_coords) / min(len(coords), len(existing_coords))
            if overlap > overlap_threshold:
                should_add = False
                break
        
        if should_add:
            final_masks[coords_set] = current_label
            current_label += 1
    
    for coords_set, label in final_masks.items():
        for x, y in coords_set:
            if 0 <= y < final_mask.shape[0] and 0 <= x < final_mask.shape[1]:
                final_mask[y, x] = 255
                final_mask_label[y, x] = label

    predicted_mask = final_mask[:original_height, :original_width]
    predicted_mask_label = final_mask_label[:original_height, :original_width]

    return predicted_mask, predicted_mask_label



def initialize_models(yolo_weights_path, unet_model_path, checkpoint):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    yolo_model = YOLO(yolo_weights_path)
    yolo_model.model = checkpoint['yolo']

    unet = unet_model.load_model_from_config(unet_model_path)
    unet.load_state_dict(checkpoint['unet'])
    unet = unet.to(device)  

    return yolo_model, unet

def update_parameters(params_file, new_values, changes=None):
    if changes is None:
        changes = {}

    with open(params_file, 'r') as yaml_file:
        params = yaml.safe_load(yaml_file)

    def recursive_update(d, new_vals, change_dict):
        for key, value in new_vals.items():
            if isinstance(value, dict):
                if key not in d:
                    d[key] = {}
                recursive_update(d[key], value, change_dict.setdefault(key, {}))
            else:
                old_value = d.get(key)
                if old_value != value:
                    change_dict[key] = {'old': old_value, 'new': value}
                    d[key] = value

    recursive_update(params, new_values, changes)
    
    return params, changes

def get_predict_results(whole_image,yolo_model,unet,config,device=None):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    original_width, original_height = whole_image.size
    patch_size = config['img_pre_process']["patch_size"]
    overlap = config['img_pre_process']["overlap"]
    padding = config['img_pre_process']["padding"]

    padded_image, left_pad, bottom_pad = pad_image_right_bottom(whole_image, patch_size)
    total_patches, total_patches_unpad, total_positions = slice_image_with_padding(
        padded_image, patch_size, overlap, padding=padding)

    print(len(total_patches))
    results=[]
    for ss,patch in enumerate(total_patches):
        
        result = yolo_model.predict(
            source=patch,
            **config['yolo_args']
        )
        results.extend(result)
    unet.eval()


    save_results = []
    final_result = []
    # print("ss1")
    for idx, patch in enumerate(total_patches):
        boxes = results[idx].boxes.data if results[idx].boxes is not None else None
        masks = results[idx].masks.data if results[idx].masks is not None else None

        if masks is None or boxes is None:
            # print(f"Warning: No mask found for patch {idx}, skipping...")
            continue
        if masks is not None:
            final_yolo_mask_total=[]
            for mask in masks.cpu():
                
                final_yolo_mask_per = resize_and_unpad(mask.unsqueeze(0).unsqueeze(0), padding=padding,size=patch.size).squeeze(0)
                final_yolo_mask_total.append(final_yolo_mask_per) 

        save_results.append((patch, total_patches_unpad[idx], total_positions[idx], boxes, final_yolo_mask_total))
    # print(len(save_results))
    # print("ss2")
    for save_result in save_results:
        patch, patch_unpads, positions, boxes, final_yolo_mask_total = save_result
        for item_bbox,item_mask in zip(boxes,final_yolo_mask_total):
            unet_mask = generate_unet_mask(patch, item_bbox.unsqueeze(0), 
                                           unet, device, 
                                           pad_size=config['unet_params']["unet_pad_size"], 
                                           model_threshold=config['unet_params']["unet_model_threshold"], 
                                           unet_img_size=config['unet_params']["unet_img_size"])
            unet_masks_cat = np.stack(unet_mask, axis=0) 
            max_mask = np.max(unet_masks_cat, axis=0)  
            unet_mask = max_mask[padding:-padding, padding:-padding]

            yolo_mask_np = np.array((item_mask[0] > 0)).astype(bool)
            intersection_mask = unet_mask & yolo_mask_np
            final_patch_mask = (intersection_mask * 255).astype(np.uint8)
            final_result.append((patch, patch_unpads, positions, final_patch_mask))
    # return padded_image,final_result,patch_size,original_height,original_width, overlap_threshold
    # print("ss3")
    predicted_mask, predicted_mask_label = process_patch_masks(padded_image,
                                                                final_result,
                                                                patch_size,
                                                                original_height,
                                                                original_width,
                                                                overlap_threshold=config['img_post_process']['overlap_threshold']
                                                                )
    return predicted_mask, predicted_mask_label

def get_total_results(whole_image,padded_image,final_result,config):

    original_width, original_height = whole_image.size
    predicted_mask, predicted_mask_label = process_patch_masks(padded_image,
                                                            final_result,
                                                            patch_size=config['img_pre_process']["patch_size"],
                                                            original_height=original_height,
                                                            original_width=original_width,
                                                            overlap_threshold=config['img_post_process']['overlap_threshold']
                                                            )
    return predicted_mask,predicted_mask_label


def get_unet_masks(save_results,unet,config,device=None):
    
    padding=config['img_pre_process']["padding"]

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    final_result=[]
    unet.eval()
    for save_result in save_results:
        patch, patch_unpads, positions, boxes, final_yolo_mask_total = save_result
        for item_bbox,item_mask in zip(boxes,final_yolo_mask_total):
            
            unet_mask = generate_unet_mask(patch, item_bbox.unsqueeze(0), 
                                           unet, device, 
                                           pad_size=config['unet_params']["unet_pad_size"], 
                                           model_threshold=config['unet_params']["unet_model_threshold"], 
                                           unet_img_size=config['unet_params']["unet_img_size"])
            unet_masks_cat = np.stack(unet_mask, axis=0) 
            max_mask = np.max(unet_masks_cat, axis=0)  
            unet_mask = max_mask[padding:-padding, padding:-padding]

            yolo_mask_np = np.array((item_mask[0] > 0)).astype(bool)
            intersection_mask = unet_mask & yolo_mask_np
            final_patch_mask = (intersection_mask * 255).astype(np.uint8)
            final_result.append((patch, patch_unpads, positions, final_patch_mask))
    return final_result


def get_yolo_predict_results(whole_image,yolo_model,config):

    original_width, original_height = whole_image.size
    patch_size = config['img_pre_process']["patch_size"]
    overlap = config['img_pre_process']["overlap"]
    padding = config['img_pre_process']["padding"]

    padded_image, left_pad, bottom_pad = pad_image_right_bottom(whole_image, patch_size)
    total_patches, total_patches_unpad, total_positions = slice_image_with_padding(
        padded_image, patch_size, overlap, padding=padding)

    # print(len(total_patches))
    results=[]
    for ss,patch in enumerate(total_patches):
        
        result = yolo_model.predict(
            source=patch,
            **config['yolo_args']
        )
        results.extend(result)

    save_results = []
    for idx, patch in enumerate(total_patches):
        boxes = results[idx].boxes.data if results[idx].boxes is not None else None
        masks = results[idx].masks.data if results[idx].masks is not None else None

        if masks is None or boxes is None:
            # print(f"Warning: No mask found for patch {idx}, skipping...")
            continue
        if masks is not None:
            final_yolo_mask_total=[]
            for mask in masks.cpu():
                final_yolo_mask_per = resize_and_unpad(mask.unsqueeze(0).unsqueeze(0), padding=padding,size=patch.size).squeeze(0)
                final_yolo_mask_total.append(final_yolo_mask_per) 

        save_results.append((patch, total_patches_unpad[idx], total_positions[idx], boxes, final_yolo_mask_total))
    return save_results,padded_image

def draw_all_regions_with_boundaries(whole_image: Image, predicted_mask_label: np.ndarray, region_color=(255, 0, 0), boundary_color=(0, 255, 0), boundary_thickness=2):
    """
    Args:
        whole_image (PIL.Image)
        predicted_mask_label (np.ndarray)
        region_color (tuple)
        boundary_color (tuple)
        boundary_thickness (int)
    Returns:
        PIL.Image:
    """
    image_np = np.array(whole_image)
    unique_labels = np.unique(predicted_mask_label)
    unique_labels = unique_labels[unique_labels != 0]

    region_colors = [(255, 0, 255), (255, 0, 255), (255, 0, 255)]  #
    boundary_colors = [(255, 0, 255), (255, 0, 255), (255, 0, 255)]  #
    for i, label in enumerate(unique_labels):
        region_color = region_colors[i % len(region_colors)]
        boundary_color = boundary_colors[i % len(boundary_colors)]
        
        mask = (predicted_mask_label == label).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image_np, contours, -1, region_color, thickness=cv2.FILLED)
        cv2.drawContours(image_np, contours, -1, boundary_color, thickness=boundary_thickness)

    return Image.fromarray(image_np)

def visualize_random_masks(whole_image, predicted_mask_label, draw_b=False,boundary_thickness=2):
    image_np = np.array(whole_image)
    
    unique_labels = np.unique(predicted_mask_label)
    
    unique_labels = unique_labels[unique_labels != 0]

    def random_color():
        return tuple(np.random.randint(0, 256, size=3))

    for i, label in enumerate(unique_labels):
        region_color = random_color()  
        boundary_color = (0,0,0)  
        mask = (predicted_mask_label == label).astype(np.uint8)
        
        kernel = np.ones((2, 2), np.uint8)
        predicted_mask_1 = cv2.dilate(mask, kernel, iterations=1)
        
        contours, _ = cv2.findContours(predicted_mask_1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        region_color = (int(region_color[0]), int(region_color[1]), int(region_color[2]))
        boundary_color = (int(boundary_color[0]), int(boundary_color[1]), int(boundary_color[2]))
        
        
        cv2.drawContours(image_np, contours, -1, region_color, thickness=cv2.FILLED)
        
        if draw_b:
            cv2.drawContours(image_np, contours, -1, boundary_color, thickness=boundary_thickness)

    return Image.fromarray(image_np)




def concat_images_vertically(images):
    # Get the width and height of the first image
    width, height = images[0].size
    
    # Calculate the total height (sum of all image heights) and the maximum width
    total_height = sum(image.size[1] for image in images)
    max_width = max(image.size[0] for image in images)
    
    # Create a new image with the calculated width and height
    new_image = Image.new("RGB", (max_width, total_height))
    
    # Paste all the images into the new image
    current_y = 0
    for image in images:
        new_image.paste(image, (0, current_y))
        current_y += image.size[1]
    
    return new_image

def concat_images_horizontally(images):
    # Get the width and height of the first image
    width, height = images[0].size
    
    # Calculate the total width (sum of all image widths) and the maximum height
    total_width = sum(image.size[0] for image in images)
    max_height = max(image.size[1] for image in images)
    
    # Create a new image with the calculated width and height
    new_image = Image.new("RGB", (total_width, max_height))
    
    # Paste all the images into the new image
    current_x = 0
    for image in images:
        new_image.paste(image, (current_x, 0))
        current_x += image.size[0]
    
    return new_image

def get_unet_masks_EL(save_results,unet,config,device=None,operation=2):
    
    padding=config['img_pre_process']["padding"]

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    final_result=[]
    unet.eval()
    for save_result in save_results:
        patch, patch_unpads, positions, boxes, final_yolo_mask_total = save_result
        for item_bbox,item_mask in zip(boxes,final_yolo_mask_total):
            unet_mask = generate_unet_mask_EL(patch=patch, 
                                          bboxes=item_bbox.unsqueeze(0), 
                                          unet=unet, 
                                          device=device, 
                                          config=config)
            
            unet_masks_cat = np.stack(unet_mask, axis=0) 
            max_mask = np.max(unet_masks_cat, axis=0)  
            unet_mask = max_mask[padding:-padding, padding:-padding]

            yolo_mask_np = np.array((item_mask[0] > 0)).astype(bool)
            if operation ==1:
                intersection_mask = unet_mask | yolo_mask_np
            else:
                intersection_mask = unet_mask & yolo_mask_np
            final_patch_mask = (intersection_mask * 255).astype(np.uint8)
            final_result.append((patch, patch_unpads, positions, final_patch_mask))
    return final_result


def pad_to_square_with_extra_pad(x_min, x_max, y_min, y_max, image_shape):
    img_h, img_w = image_shape
    
    width = x_max - x_min
    height = y_max - y_min

    max_side = max(width, height)

    pad_w = (max_side - width) / 2
    pad_h = (max_side - height) / 2

    pad_left = int(pad_w)
    pad_right = int(pad_w + 0.5)  
    pad_top = int(pad_h)
    pad_bottom = int(pad_h + 0.5)

    x_min_padded = max(x_min - pad_left, 0)
    if x_min_padded == 0:  
        x_max_padded = min(x_max + pad_right + (pad_left-x_min), img_w)  
    else:
        x_max_padded = min(x_max + pad_right, img_w)  
    if x_max_padded >= img_w:
        x_min_padded = max(x_min_padded - (x_max + pad_right - img_w), 0)


    y_min_padded = max(y_min - pad_top, 0)
    if y_min_padded == 0:  
        y_max_padded = min(y_max + pad_bottom + (pad_top-y_min), img_h)
    else:
        y_max_padded = min(y_max + pad_bottom, img_h)
    if y_max_padded >= img_h:
        y_min_padded = min(y_min_padded - (y_max + pad_bottom - img_h), img_h)

    return x_min_padded, x_max_padded, y_min_padded, y_max_padded


def pad_bbox_with_adjustment(x_min, x_max, y_min, y_max, image_shape, padsize):
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

