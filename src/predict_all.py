from predict import predict

# Set your model and config paths
model_path = "/scratch/kf1d20/2024.11.18_RT_DETR/yolov11/runs/train/exp4/weights/best.pt"
model_path = "/scratch/kf1d20/2024.11.18_RT_DETR/yolov11/runs/train/exp4/weights/last.pt"
model_config_path = "/scratch/kf1d20/2024.11.18_RT_DETR/yolov11/runs/train/exp4/args.yaml"  # agnostic_nms=True in this file

# Set your images directory
images_dir = "/scratch/kf1d20/2024.11.18_RT_DETR/yolov11/test/SA TMA 1-Image Export-03_s009.jpg"
images_dir = "./test/000000000003.png"
# Run SAHI's predict function with the necessary parameters
predict(
    model_type="yolov8",
    model_path=model_path,
    model_config_path=model_config_path,
    model_device="cuda:0",  # or "cpu"
    model_confidence_threshold=0.8,
    postprocess_class_agnostic=True,
    source=images_dir,
    slice_height=640,
    slice_width=640,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
    visual_hide_conf=True,
    visual_hide_labels=True,
    visual_bbox_thickness=0
    # ... other parameters as needed
)

# project