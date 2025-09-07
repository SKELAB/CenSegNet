import torch
import cv2
import tifffile
import yaml
import numpy as np
from PIL import Image, ImageDraw
from segmentors import unet_model
from ultralytics import YOLO
import requests
from typing import List, Optional, Union


def initialize_models_all_3(yolo_weights_path, unet_model_path, checkpoint,mode="IF"):
    # 初始化 YOLO 模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    yolo_model = YOLO(yolo_weights_path)
    unet = unet_model.load_model_from_config(unet_model_path)

    if mode =="IF":
        yolo_model.model = checkpoint['IF']['yolo']
        unet.load_state_dict(checkpoint['IF']['unet'])
    elif mode =="IHC":
        yolo_model.model = checkpoint['IHC']['yolo']
        unet.load_state_dict(checkpoint['IHC']['unet'])
    elif mode =="EL":
        yolo_model.model = checkpoint['EL']['yolo']
        unet.load_state_dict(checkpoint['EL']['unet'])
    unet = unet.to(device)  # 将 UNet 模型移动到设备

    return yolo_model, unet


def init_parameters(params_file, new_values, changes=None):
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

def updated_parameters(params, new_values, changes=None):
    if changes is None:
        changes = {}

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



def init_params(params_file,mode):
    
    if mode == "Epithelial" or mode == "EL":
        new_values= {
            'img_pre_process': {"overlap": 0.35,"patch_size":580,"padding":30},
            "unet_params":{"unet_model_threshold": 0.15,"unet_pad_size":50,"unet_img_size":512,"unet_threshold":300,},
            'yolo_args': {'conf':0.01,},
            "img_post_process":{"overlap_threshold": 0.99}}
    if mode == "IF":
        new_values = {'img_pre_process': {"overlap": 0.5,},
        "unet_params":{"unet_model_threshold":0.02,"unet_pad_size":5,"unet_img_size":128,},
        'yolo_args': {'conf':0.01,},
        "img_post_process":{"overlap_threshold": 0.95}
        }
    if mode == "IHC":
        new_values = {
        'img_pre_process': {"overlap": 0.5,"patch_size":500,"padding":6},
        "unet_params":{"unet_model_threshold":0.1,"unet_pad_size":10,"unet_img_size":128,},
        'yolo_args': {'conf':0.1,},
        "img_post_process":{"overlap_threshold": 0.5}
        }
    updated_config, changes = init_parameters(params_file, new_values)

    return updated_config, changes


def read_image_as_pil(image: Union[Image.Image, str, np.ndarray], exif_fix: bool = False):
    """
    Loads an image as PIL.Image.Image.

    Args:
        image (Union[Image.Image, str, np.ndarray]): The image to be loaded. It can be an image path or URL (str),
            a numpy image (np.ndarray), or a PIL.Image object.
        exif_fix (bool, optional): Whether to apply an EXIF fix to the image. Defaults to False.

    Returns:
        PIL.Image.Image: The loaded image as a PIL.Image object.
    """
    # https://stackoverflow.com/questions/56174099/how-to-load-images-larger-than-max-image-pixels-with-pil
    Image.MAX_IMAGE_PIXELS = None

    if isinstance(image, Image.Image):
        image_pil = image
    elif isinstance(image, str):
        # read image if str image path is provided
        try:
            image_pil = Image.open(
                requests.get(image, stream=True).raw if str(image).startswith("http") else image
            ).convert("RGB")
            if exif_fix:
                image_pil = exif_transpose(image_pil)
        except:  # handle large/tiff image reading
            try:
                import skimage.io
            except ImportError:
                raise ImportError("Please run 'pip install -U scikit-image imagecodecs' for large image handling.")
            image_sk = skimage.io.imread(image).astype(np.uint8)
            if len(image_sk.shape) == 2:  # b&w
                image_pil = Image.fromarray(image_sk, mode="1")
            elif image_sk.shape[2] == 4:  # rgba
                image_pil = Image.fromarray(image_sk, mode="RGBA")
            elif image_sk.shape[2] == 3:  # rgb
                image_pil = Image.fromarray(image_sk, mode="RGB")
            else:
                raise TypeError(f"image with shape: {image_sk.shape[3]} is not supported.")
    elif isinstance(image, np.ndarray):
        if image.shape[0] < 5:  # image in CHW
            image = image[:, :, ::-1]
        image_pil = Image.fromarray(image)
    else:
        raise TypeError("read image with 'pillow' using 'Image.open()'")
    return image_pil

def exif_transpose(image: Image.Image) -> Image.Image:
    """
    Transpose a PIL image accordingly if it has an EXIF Orientation tag.
    Inplace version of https://github.com/python-pillow/Pillow/blob/master/src/PIL/ImageOps.py exif_transpose()

    Args:
        image (Image.Image): The image to transpose.

    Returns:
        Image.Image: The transposed image.
    """
    exif = image.getexif()
    orientation = exif.get(0x0112, 1)  # default 1
    if orientation > 1:
        method = {
            2: Image.FLIP_LEFT_RIGHT,
            3: Image.ROTATE_180,
            4: Image.FLIP_TOP_BOTTOM,
            5: Image.TRANSPOSE,
            6: Image.ROTATE_270,
            7: Image.TRANSVERSE,
            8: Image.ROTATE_90,
        }.get(orientation)
        if method is not None:
            image = image.transpose(method)
            del exif[0x0112]
            image.info["exif"] = exif.tobytes()
    return image

def get_image(image_file,mode="IF"):
    if mode =="IF":
        original_image = tifffile.imread(image_file)
        normalized_image = cv2.normalize(original_image[-1], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        rgb_image = cv2.cvtColor(normalized_image, cv2.COLOR_GRAY2RGB)
        whole_image = Image.fromarray(rgb_image)
    elif mode =="IHC" or mode =="EL":
        whole_image = read_image_as_pil(image_file)
    return whole_image


def generate_mask_EL_IHC(whole_image,predicted_mask,mode):

    fill_color = (193, 64, 68, 255)  # RGBA
    # whole_image = get_image(image_file, "IHC")
    if mode =="IHC":
        fill_color = (193, 64, 68, 255)  # RGBA
    if mode =="EL":
        fill_color = (122, 135, 134, 255)  # RGBA

    kernel_point = np.ones((3, 3), np.uint8)  # 用于膨胀点
    kernel_line = np.ones((5, 5), np.uint8)  # 用于膨胀线段
    kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    contours, _ = cv2.findContours(predicted_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 创建两个空白掩码，分别用于存储膨胀后的点和线段
    mask_points = np.zeros_like(predicted_mask)
    mask_lines = np.zeros_like(predicted_mask)
    mask_polygons = np.zeros_like(predicted_mask)
    for contour in contours:
        if len(contour) == 1:  # 点
            # 获取点的中心坐标
            x, y = contour[0][0]
            # 在 mask_points 上绘制一个更大的圆点
            cv2.circle(mask_points, (x, y), radius=2, color=255, thickness=-1)  # 半径为 2
        elif len(contour) == 2:  # 线段
            # 在 mask_lines 上绘制更粗的线段
            pt1 = tuple(contour[0][0])
            pt2 = tuple(contour[1][0])
            cv2.line(mask_lines, pt1, pt2, color=255, thickness=4)  # 线宽为 4
        else:  # 多边形
            # 在 mask_polygons 上绘制多边形
            cv2.drawContours(mask_polygons, [contour], -1, 255, thickness=cv2.FILLED)

    mask_points_dilated = cv2.dilate(mask_points, kernel_point, iterations=1)
    mask_lines_dilated = cv2.dilate(mask_lines, kernel_line, iterations=1)

    mask_polygons_smoothed = cv2.morphologyEx(mask_polygons, cv2.MORPH_CLOSE, kernel_smooth)  # 闭运算填充小孔洞
    mask_polygons_smoothed = cv2.morphologyEx(mask_polygons_smoothed, cv2.MORPH_OPEN, kernel_smooth)  # 开运算去除小噪声

    # 合并膨胀后的点和线段以及平滑后的多边形
    mask_combined = cv2.addWeighted(mask_points_dilated, 1, mask_lines_dilated, 1, 0)
    mask_combined = cv2.addWeighted(mask_combined, 1, mask_polygons_smoothed, 1, 0)

    # 将掩码转换为 PIL 图像
    mask_pil = Image.fromarray(mask_combined)
    mask_data = np.array(mask_pil)
    mask_filled = mask_data.copy()

    whole_image_draw = whole_image.convert("RGBA")


    dpi = 300
    stroke_width = int(0.1 * dpi / 25.4)  # 0.1mm 转换为像素
    
    overlay = Image.new("RGBA", whole_image.size, (0, 0, 0, 0))
    overlay_data = np.array(overlay)
    nonzero_mask = mask_filled > 0
    overlay_data[nonzero_mask] = fill_color  # 赋值填充颜色
    overlay_pil = Image.fromarray(overlay_data, mode="RGBA")
    draw = ImageDraw.Draw(overlay_pil)

    contours, _ = cv2.findContours(mask_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        contour = [(point[0][0], point[0][1]) for point in contour]  # 转换格式
        if len(contour) == 1:
            # 绘制单点：画一个小圆点
            x, y = contour[0]
            draw.ellipse([x - 1, y - 1, x + 1, y + 1], fill=(0, 0, 0, 255))
        elif len(contour) == 2:
            # 绘制线段
            draw.line([(contour[0][0], contour[0][1]), (contour[1][0], contour[1][1])], fill=(0, 0, 0, 255), width=1)
        else:    
            # 绘制多边形
            
            draw.polygon(contour, outline=(0, 0, 0, 255), width=1)

    # 3. **叠加 Mask 到原始图像**
    result = Image.alpha_composite(whole_image_draw, overlay_pil)

    return result,overlay_pil


def generate_mask_IF(whole_image,mask,type=1):
    if type ==1:
        fill_color = (193, 64, 68, 255)  # RGBA
    else:
        fill_color = (0, 255, 0, 255)  # RGBA

    whole_image_draw = whole_image.convert("RGBA")

    kernel = np.ones((2, 2), np.uint8) 
    pred_mask_2 = cv2.dilate(mask, kernel, iterations=1)
    pred_mask = (pred_mask_2 == 255).astype(np.uint8)

    mask = np.where(pred_mask > 0, 255, 0).astype(np.uint8)
    overlay = Image.new("RGBA", whole_image.size, (0, 0, 0, 0))
    overlay_data = np.array(overlay)
    nonzero_mask = mask > 0
    overlay_data[nonzero_mask] = fill_color  # 赋值填充颜色
    overlay_pil = Image.fromarray(overlay_data, mode="RGBA")

    result = Image.alpha_composite(whole_image_draw, overlay_pil)
    
    
    return result,overlay_pil


def plot_full_label_mask_pil(predicted_mask_label: np.ndarray):
    """
    使用 PIL 绘制完整的 predicted_mask_label 灰度图。

    Args:
        predicted_mask_label (np.ndarray): 预测的标签掩码。
    """
    # 归一化到 0-255 之间
    normalized_mask = (predicted_mask_label - predicted_mask_label.min()) / \
                      (predicted_mask_label.max() - predicted_mask_label.min()) * 255
    normalized_mask = normalized_mask.astype(np.uint8)

    # 创建 PIL 图像并显示
    img = Image.fromarray(normalized_mask, mode='L')
    return img


