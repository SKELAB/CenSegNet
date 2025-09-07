import os
import argparse
import copy
import numpy as np
import pandas as pd
import cv2
import tifffile
from PIL import Image
import torch

from GUI_tools import (
    initialize_models_all_3, init_params, updated_parameters,
    get_image, generate_mask_EL_IHC, generate_mask_IF, plot_full_label_mask_pil
)
from tools import (
    get_unet_masks, get_unet_masks_EL,
    get_total_results, get_yolo_predict_results
)


def load_models(args):
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    mode = "EL" if args.mode == "Epithelial" else args.mode

    yolo, unet = initialize_models_all_3(
        yolo_weights_path=args.yolo_model_path,
        unet_model_path=args.unet_model_path,
        checkpoint=torch.load(args.checkpoint, map_location="cuda" if torch.cuda.is_available() else "cpu"),
        mode=mode,
    )

    config, _ = init_params(args.predict_config, args.mode)
    return yolo, unet, config


def load_image(path, mode):
    if mode == "IF":
        original_image = tifffile.imread(path)
        target_channel = original_image[-1]
        img_uint8 = cv2.normalize(target_channel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        img_rgb = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(img_rgb)
    else:  # IHC / EL / Epithelial
        return get_image(path, "IHC")


def run_prediction(image, yolo, unet, config, mode, overlap, unet_thresh, yolo_thresh):
    # Update config thresholds
    updated_params = {
        "img_pre_process": {"overlap": overlap},
        "unet_params": {"unet_model_threshold": unet_thresh},
        "yolo_args": {"conf": yolo_thresh},
    }
    config, _ = updated_parameters(config, updated_params)

    # YOLO
    yolo_results, padded_img = get_yolo_predict_results(image, yolo, config)

    # UNet
    if mode in ["EL", "Epithelial"]:
        unet_results = get_unet_masks_EL(yolo_results, unet, config)
    else:
        unet_results = get_unet_masks(yolo_results, unet, config)

    # Final results
    predicted_mask, rois_map = get_total_results(image, padded_image=padded_img, final_result=unet_results, config=config)
    mask_label_pil = plot_full_label_mask_pil(rois_map)

    # Extract bounding boxes
    num_labels, labels = cv2.connectedComponents(np.array(mask_label_pil))
    bboxes = []
    for label in range(1, num_labels):
        points = np.column_stack(np.where(labels == label))
        x, y, w, h = cv2.boundingRect(points)
        bboxes.append((x, y, w, h, len(points)))

    df = pd.DataFrame(bboxes, columns=["x", "y", "w", "h", "num_points"])

    # Overlay mask
    if mode in ["EL", "IHC", "Epithelial"]:
        u_mode = "EL" if mode == "Epithelial" else mode
        masked_img, _ = generate_mask_EL_IHC(image, predicted_mask, u_mode)
    else:
        masked_img, _ = generate_mask_IF(image, predicted_mask)

    return masked_img, df


def main():
    parser = argparse.ArgumentParser(description="CenSegNet CLI - Centrosome Segmentation")
    parser.add_argument("--mode", type=str, choices=["IF", "IHC", "Epithelial"], default="IF")
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--yolo_model_path", type=str, default="./weights/yolov11/yolo11m-seg.pt")
    parser.add_argument("--unet_model_path", type=str, default="./configs/unet-config.yaml")
    parser.add_argument("--predict_config", type=str, default="./configs/predict.yaml")
    parser.add_argument("--outdir", type=str, default="./results")
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--seg_thresh", type=float, default=0.15)
    parser.add_argument("--det_thresh", type=float, default=0.15)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("Loading models...")
    yolo, unet, config = load_models(args)

    print("Loading image...")
    image = load_image(args.image, args.mode)

    print("Running prediction...")
    masked_img, df = run_prediction(
        image, yolo, unet, config, args.mode,
        overlap=args.overlap, unet_thresh=args.seg_thresh, yolo_thresh=args.det_thresh
    )

    # Save outputs
    base = os.path.splitext(os.path.basename(args.image))[0]
    masked_img.save(os.path.join(args.outdir, f"{base}_result.png"))
    df.to_csv(os.path.join(args.outdir, f"{base}_data.csv"), index=False)

    print(f"Done! Saved results to {args.outdir}")


if __name__ == "__main__":
    main()
