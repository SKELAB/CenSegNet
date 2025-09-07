import numpy as np
import torch
import cv2
from torchvision import transforms
from PIL import Image


def generate_unet_mask_EL(patch, bboxes, unet, device, config):

    image_size =image_shape= patch.size
    
    total_mask_results=[]
    patch_array=np.array(patch)

    target_transform = transforms.Compose([
        transforms.Resize((config['unet_params']["unet_img_size"], config['unet_params']["unet_img_size"])),
        transforms.ToTensor() 
    ])
    unet.eval()

    for box in bboxes:
        unet_mask = np.zeros(image_size[::-1], dtype=bool)  

        x_min_o, y_min_o, x_max_o, y_max_o, _, _ = box
        # print(box)

        x_min_padded, x_max_padded, y_min_padded, y_max_padded = pad_to_square_with_extra_pad(int(x_min_o), int(x_max_o), 
                                                                                            int(y_min_o), int(y_max_o), image_shape)
            
        if x_max_padded -  x_min_padded <=config['unet_params']["unet_threshold"]:
            x_min_padded, x_max_padded, y_min_padded, y_max_padded= pad_bbox_with_adjustment(x_min_padded, x_max_padded, y_min_padded, y_max_padded, 
                                                                                            image_shape, padsize=config['unet_params']["unet_pad_size"])
        
        cropped_image_raw = patch_array[y_min_padded:y_max_padded, x_min_padded:x_max_padded]
        # cropped_image_norm = (cropped_image_raw - cropped_image_raw.min()) / (cropped_image_raw.max() - cropped_image_raw.min()) * 255  
        cropped_image = Image.fromarray(cropped_image_raw.astype(np.uint8))

        input_tensor = target_transform(cropped_image).unsqueeze(0)
        with torch.no_grad():
            output = unet(input_tensor.to(device))
            sigmoid = torch.sigmoid(output)

        normalized_tensor = (sigmoid - sigmoid.min()) / (sigmoid.max() - sigmoid.min())
        predicted_mask = normalized_tensor.squeeze(0).squeeze(0).cpu().numpy()

        binary_mask = (predicted_mask > config['unet_params']["unet_model_threshold"]).astype(np.float32)
        
        binary_mask_image = cv2.resize(binary_mask, (x_max_padded - x_min_padded, y_max_padded - y_min_padded), 
                                       interpolation=cv2.INTER_LINEAR)
        
        unet_mask[y_min_padded:y_max_padded, x_min_padded:x_max_padded] = binary_mask_image.astype(bool)
        total_mask_results.append(unet_mask)

    return total_mask_results




def generate_unet_mask(patch, bboxes, unet, device, pad_size=20, model_threshold=0.10, unet_img_size=128, return_position_info=False):
    """
    Process image `patch` and `bboxes`, use UNet to predict masks, 
    and return the composite binary mask.

    Args:
        patch (PIL.Image): Input image.
        bboxes (List[Tuple]): List of detection boxes, 
            each box in format (x_min, y_min, x_max, y_max, score, class_id).
        unet (torch.nn.Module): Preloaded UNet model.
        device (torch.device): PyTorch device (cuda / cpu).
        pad_size (int): Number of pixels to expand bbox.
        model_threshold (float): Threshold for binarizing predicted mask.
        unet_img_size (int): Input size for UNet.
        return_position_info (bool): Whether to also return bbox info.

    Returns:
        - list[np.ndarray]: Binary masks (cropped and padded).
        - (optional) List of bboxes if return_position_info=True.
    """
    # Get image size
                        
    image_size = patch.size  # (width, height)
    
    # Collect masks
    total_mask_results = []
    if return_position_info:
        total_return_info = []
    
    # Preprocessing transform
    target_transform = transforms.Compose([
        transforms.Resize((unet_img_size, unet_img_size)),  # Resize for model
        transforms.ToTensor(),  # Convert to tensor
    ])

    # Ensure UNet is in eval mode
    unet.eval()

    for box in bboxes:
        unet_mask = np.zeros(image_size[::-1], dtype=bool)  # (H, W), need reversed size

        x_min, y_min, x_max, y_max, _, _ = box

        # Expand bbox and ensure within image boundaries
        x_min = int(max(x_min - pad_size, 0))
        y_min = int(max(y_min - pad_size, 0))
        x_max = int(min(x_max + pad_size, image_size[0]))  # width
        y_max = int(min(y_max + pad_size, image_size[1]))  # height

        # Crop image
        patch_array = np.array(patch)
        cropped_image = Image.fromarray(patch_array[y_min:y_max, x_min:x_max])

        # Preprocess input
        input_tensor = target_transform(cropped_image).unsqueeze(0)

        # Run UNet prediction
        with torch.no_grad():
            output = unet(input_tensor.to(device))
            sigmoid = torch.sigmoid(output)

        # Normalize
        normalized_tensor = (sigmoid - sigmoid.min()) / (sigmoid.max() - sigmoid.min())
        predicted_mask = normalized_tensor.squeeze(0).squeeze(0).cpu().numpy()

        # Binarize
        binary_mask = (predicted_mask > model_threshold).astype(np.float32)

        # Resize back to bbox size
        if x_max > x_min and y_max > y_min:
            binary_mask_image = cv2.resize(binary_mask, (x_max - x_min, y_max - y_min), interpolation=cv2.INTER_LINEAR)
        else:
            # Handle invalid bbox
            padding = 2
            x_max += padding
            x_min -= padding
            y_max += padding
            y_min -= padding

            x_min = max(x_min, 0)
            y_min = max(y_min, 0)
            x_max = min(x_max, image_size[0])
            y_max = min(y_max, image_size[1])

            print(f"Invalid bbox, added padding: x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}")
            binary_mask_image = cv2.resize(binary_mask, (x_max - x_min, y_max - y_min), interpolation=cv2.INTER_LINEAR)

        # Merge into global mask
        unet_mask[y_min:y_max, x_min:x_max] = binary_mask_image.astype(bool)

        total_mask_results.append(unet_mask)
        if return_position_info:
            total_return_info.append(box)

    if return_position_info:
        return total_return_info, total_mask_results
    else:
        return total_mask_results

def generate_unet_mask_old(patch, bboxes, unet, device, pad_size=20, model_threshold=0.10, unet_img_size=128):
    image_size = patch.size  # (width, height)
    
    unet_mask = np.zeros(image_size[::-1], dtype=bool)
    
    target_transform = transforms.Compose([
        transforms.Resize((unet_img_size, unet_img_size)),
        transforms.ToTensor(),  
    ])

    unet.eval()

    for box in bboxes:
        x_min, y_min, x_max, y_max, _, _ = box

        x_min = int(max(x_min - pad_size, 0))
        y_min = int(max(y_min - pad_size, 0))
        x_max = int(min(x_max + pad_size, image_size[0]))  # width
        y_max = int(min(y_max + pad_size, image_size[1]))  # height

        patch_array = np.array(patch)  
        cropped_image = Image.fromarray(patch_array[y_min:y_max, x_min:x_max])

        input_tensor = target_transform(cropped_image).unsqueeze(0)

        with torch.no_grad():
            output = unet(input_tensor.to(device))
            sigmoid = torch.sigmoid(output)

        normalized_tensor = (sigmoid - sigmoid.min()) / (sigmoid.max() - sigmoid.min())
        predicted_mask = normalized_tensor.squeeze(0).squeeze(0).cpu().numpy()

        binary_mask = (predicted_mask > model_threshold).astype(np.float32)

        if x_max > x_min and y_max > y_min:
            binary_mask_image = cv2.resize(binary_mask, (x_max - x_min, y_max - y_min), interpolation=cv2.INTER_LINEAR)
        else:
            padding=2
            x_max += padding
            x_min -= padding
            y_max += padding
            y_min -= padding

            x_min = max(x_min, 0)
            y_min = max(y_min, 0)
            x_max = min(x_max, image_size[0])
            y_max = min(y_max, image_size[1])

            print(f"Invalid bbox, added padding: x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}")
            binary_mask_image = cv2.resize(binary_mask, (x_max - x_min, y_max - y_min), interpolation=cv2.INTER_LINEAR)

        unet_mask[y_min:y_max, x_min:x_max] = binary_mask_image.astype(bool)


    return unet_mask

def apply_color_mask(image: np.ndarray, color: tuple):
    """
    Applies color mask to given input image.

    Args:
        image (np.ndarray): The input image to apply the color mask to.
        color (tuple): The RGB color tuple to use for the mask.

    Returns:
        np.ndarray: The resulting image with the applied color mask.
    """
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    (r[image == 1], g[image == 1], b[image == 1]) = color
    colored_mask = np.stack([r, g, b], axis=2)
    return colored_mask

class Colors:
    def __init__(self):
        hex = (
            "FF3838",
            "2C99A8",
            "FF701F",
            "6473FF",
            "CFD231",
            "48F90A",
            "92CC17",
            "3DDB86",
            "1A9334",
            "00D4BB",
            "FF9D97",
            "00C2FF",
            "344593",
            "FFB21D",
            "0018EC",
            "8438FF",
            "520085",
            "CB38FF",
            "FF95C8",
            "FF37C7",
        )
        self.palette = [self.hex_to_rgb("#" + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, ind, bgr: bool = False):
        """
        Convert an index to a color code.

        Args:
            ind (int): The index to convert.
            bgr (bool, optional): Whether to return the color code in BGR format. Defaults to False.

        Returns:
            tuple: The color code in RGB or BGR format, depending on the value of `bgr`.
        """
        color_codes = self.palette[int(ind) % self.n]
        return (color_codes[2], color_codes[1], color_codes[0]) if bgr else color_codes

    @staticmethod
    def hex_to_rgb(hex_code):
        """
        Converts a hexadecimal color code to RGB format.

        Args:
            hex_code (str): The hexadecimal color code to convert.

        Returns:
            tuple: A tuple representing the RGB values in the order (R, G, B).
        """
        rgb = []
        for i in (0, 2, 4):
            rgb.append(int(hex_code[1 + i : 1 + i + 2], 16))
        return tuple(rgb)


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




