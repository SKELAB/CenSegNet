import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QGroupBox, QRadioButton, QLineEdit, QSlider,
    QScrollArea, QButtonGroup, QFrame,QMessageBox,QTextEdit,QGridLayout,QSizePolicy
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QCursor
from PyQt5.QtGui import QIcon # 20250610, JC, for adding a new icon
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import copy
import cv2
import tifffile
from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage,QFont
import torch
import threading
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from GUI_tools import initialize_models_all_3, init_params, updated_parameters, get_image
from GUI_tools import generate_mask_EL_IHC, generate_mask_IF,plot_full_label_mask_pil
from tools import (get_unet_masks, get_unet_masks_EL, get_total_results, get_yolo_predict_results)


class ImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._pixmap = None
        self._original_pixmap = None
        self.zoom_factor = 1.0

    def setPixmapOriginal(self, pixmap):
        self._original_pixmap = pixmap
        self._pixmap = pixmap
        super().setPixmap(pixmap.scaled(
            self.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        ))

    def reset_view(self):
        if self._original_pixmap:
            self.zoom_factor = 1.0
            self.setPixmapOriginal(self._original_pixmap)

    def resizeEvent(self, event):
        if self._pixmap:
            super().setPixmap(self._pixmap.scaled(
                self.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            ))
        super().resizeEvent(event)


class MainWindow(QMainWindow):
    # Add signals for inter-thread communication
    prediction_finished = pyqtSignal(int)  # Parameter is ROI count
    prediction_failed = pyqtSignal(str)   # Parameter is error message
    update_status = pyqtSignal(str)       # Update status text
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CenSegNet")
        self.setWindowIcon(QIcon("./utils/newicon.png")) # 20250610, JC, for adding a new icon
        self.setGeometry(100, 100, 1200, 800)

        self.args = self.init_args()

        self.yolo = None
        self.unet = None
        self.config = None

        self.sliders = {}
        self.slider_info = {
            "Overlap Ratio": {
                "range": (0, 100),
                "default": 50,
                "format": lambda v: f"{v / 100:.2f}",
                "key": "overlap"
            },
            "Detection Threshold": {
                "range": (0, 100),
                "default": 15,
                "format": lambda v: f"{v / 100:.2f}",
                "key": "detection"
            },
            "Segmentation Threshold": {
                "range": (0, 100),
                "default": 15,
                "format": lambda v: f"{v / 100:.2f}",
                "key": "segmentation"
            }
        }

        # Add control variables
        self.is_predicting = False
        self.should_stop = False
        self.predict_thread = None
        
        self.models_loaded = False  # Initial state is not loaded
        self.current_mode = None   # Record currently loaded mode

        self.initUI()

        # Connect signals
        self.prediction_finished.connect(self.on_prediction_finished)
        self.prediction_failed.connect(self.on_prediction_failed)
        self.update_status.connect(self.label_rois.setText)

    def initUI(self):
        # Main container setup
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        # Main layout configuration
        main_layout = QGridLayout(main_widget)
        main_layout.setColumnStretch(0, 3)  # Left panel
        main_layout.setColumnStretch(1, 7)  # Right panel
        main_layout.setRowStretch(0, 2)     # Header row
        main_layout.setRowStretch(1, 8)     # Content row
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)

        # Header section with logo and title
        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)

        # Logo setup
        logo_label = QLabel()
        logo_pixmap = QPixmap("logo.png").scaled(150, 150, Qt.KeepAspectRatio)
        logo_label.setPixmap(logo_pixmap)

        # Title with larger font
        title_label = QLabel("CenSegNet: Centrosome Segmentation")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            QLabel {
                font: bold 32px Arial !important;
            }
        """)
        
        header_layout.addWidget(logo_label)
        header_layout.addStretch()
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        main_layout.addWidget(header_widget, 0, 0, 1, 2)

        # Left control panel
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(5, 5, 5, 5)
        left_layout.setSpacing(8)

        # 2. Image upload section
        image_group = QGroupBox("Upload image")
        font = QFont("Microsoft YaHei", 18)#, QFont.Bold)
        image_group.setFont(font)
        image_layout = QVBoxLayout()
        image_layout.setSpacing(6)

        self.radio_buttons = QButtonGroup(self)
        for mode in ["IF", "IHC", "Epithelial"]:
            hbox = QHBoxLayout()
            radio = QRadioButton(mode)
            if mode == self.args.mode:
                radio.setChecked(True)
            radio.toggled.connect(lambda checked, m=mode: self.update_mode_and_reset_display(checked, m))
            
            browse = QPushButton("Browse")
            browse.clicked.connect(lambda _, m=mode: self.select_image(m))
            hbox.addWidget(radio)
            hbox.addWidget(browse)
            image_layout.addLayout(hbox)
            self.radio_buttons.addButton(radio)
        '''
        # Selected image display
        selected_image_group = QGroupBox("Selected Image Path")
        selected_image_layout = QVBoxLayout()
        self.selected_image_display = QLabel("NO Image Selected")
        self.selected_image_display.setWordWrap(True)
        self.selected_image_display.setMinimumHeight(80)
        self.selected_image_display.setStyleSheet("""
            QLabel {
                font: 12px ;
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 8px;
                color: #495057;
            }
        """)
        selected_image_layout.addWidget(self.selected_image_display)
        selected_image_group.setLayout(selected_image_layout)
        image_layout.addWidget(selected_image_group)
        '''
        
                
                
        # Selected image display
        # selected_image_group = QGroupBox("Selected Image Path")
        selected_image_layout = QVBoxLayout()
        self.selected_image_display = QLabel("NO Image Selected")
        self.selected_image_display.setWordWrap(True)
        self.selected_image_display.setMinimumHeight(60)
        self.selected_image_display.setStyleSheet("""
            QLabel {
                font: 12px ;
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 8px;
                color: #495057;
            }
        """)
        selected_image_layout.addWidget(self.selected_image_display)
        # selected_image_group.setLayout(selected_image_layout)
        image_layout.addLayout(selected_image_layout)
                
        
        btn_upload = QPushButton("Confirm Selected Image")
        btn_upload.clicked.connect(self.upload_image)
        image_layout.addWidget(btn_upload)
        image_group.setLayout(image_layout)
        left_layout.addWidget(image_group)

        # 1. Model upload section
        btn_upload_ckpt = QPushButton("Upload CenSegNet (Checkpoint)")
        btn_upload_ckpt.clicked.connect(self.upload_checkpoint)
        btn_upload_ckpt.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        left_layout.addWidget(btn_upload_ckpt)

        # 3. Prediction section
        predict_group = QGroupBox("Prediction")
        font = QFont("Microsoft YaHei", 18)#, QFont.Bold)
        predict_group.setFont(font)
        predict_layout = QVBoxLayout()
        predict_layout.setSpacing(6)

        # Slider controls
        slider_container = QWidget()
        slider_layout = QVBoxLayout(slider_container)
        slider_layout.setSpacing(4)
        
        slider_font = QFont("Microsoft YaHei", 10)  # 20250610 JC, font #QFont("Arial", 6)#, QFont.Bold)

        for name, config in self.slider_info.items():
            box = QVBoxLayout()
            box.setSpacing(4)
            
            label = QLabel(f"{name}: {config['format'](config['default'])}")
            label.setFont(slider_font)
            slider = QSlider(Qt.Horizontal)
            slider.setRange(*config['range'])
            slider.setValue(config['default'])
            
            slider.valueChanged.connect(
                lambda v, l=label, n=name, f=config['format']: l.setText(f"{n}: {f(v)}")
            )
            
            box.addWidget(label)
            box.addWidget(slider)
            self.sliders[config['key']] = slider
            slider_layout.addLayout(box)

        predict_layout.addWidget(slider_container)

        self.label_rois = QLabel("Waiting Start: 0 ROIs")
        # Set font
        font = QFont("Microsoft YaHei", 10)
        self.label_rois.setFont(font)
        predict_layout.addWidget(self.label_rois)

        # Prediction buttons
        btn_container = QWidget()
        btn_layout = QHBoxLayout(btn_container)
        btn_layout.setContentsMargins(0, 0, 0, 0)
        
        btn_reset = QPushButton("Reset")
        btn_reset.clicked.connect(self.reset_sliders)
        btn_layout.addWidget(btn_reset)

        self.btn_predict = QPushButton("Predict")
        self.btn_predict.clicked.connect(self.predict)
        btn_layout.addWidget(self.btn_predict)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_prediction)
        self.btn_stop.setStyleSheet("""
            QPushButton {
                background-color: #FF4444;
                color: white;
            }
            QPushButton:disabled {
                background-color: #CCCCCC;
            }
        """)
        btn_layout.addWidget(self.btn_stop)

        predict_layout.addWidget(btn_container)
        predict_group.setLayout(predict_layout)
        left_layout.addWidget(predict_group)

        # 4. Export section
        export_group = QGroupBox("Export")
        font = QFont("Microsoft YaHei", 18)#, QFont.Bold)
        export_group.setFont(font)
        export_layout = QVBoxLayout()
        export_layout.setSpacing(6)

        # Export prediction results
        result_group = QGroupBox("Export Prediction Results")
        font = QFont("Microsoft YaHei", 14)#, QFont.Bold)
        result_group.setFont(font)
        result_layout = QVBoxLayout()
        result_layout.setContentsMargins(5, 10, 5, 5)

        result_path_layout = QHBoxLayout()
        self.line_result_path = QLineEdit()
        self.line_result_path.setPlaceholderText("Prediction output path...")
        btn_browse_result = QPushButton("Browse")
        btn_browse_result.clicked.connect(lambda: self.browse_export_path("result"))
        result_path_layout.addWidget(self.line_result_path)
        result_path_layout.addWidget(btn_browse_result)

        self.btn_export_result = QPushButton("Export Results")
        self.btn_export_result.clicked.connect(self.export_prediction_results)
        result_layout.addLayout(result_path_layout)
        result_layout.addWidget(self.btn_export_result)
        result_group.setLayout(result_layout)

        # Export ROI data
        data_group = QGroupBox("Export ROI Results")
        font = QFont("Microsoft YaHei", 14)#, QFont.Bold)
        data_group.setFont(font)
        data_layout = QVBoxLayout()
        data_layout.setContentsMargins(5, 10, 5, 5)

        data_path_layout = QHBoxLayout()
        self.line_data_path = QLineEdit()
        self.line_data_path.setPlaceholderText("Data export path...")
        btn_browse_data = QPushButton("Browse")
        btn_browse_data.clicked.connect(lambda: self.browse_export_path("data"))
        data_path_layout.addWidget(self.line_data_path)
        data_path_layout.addWidget(btn_browse_data)

        self.btn_export_data = QPushButton("Export ROI Data")
        self.btn_export_data.clicked.connect(self.export_analysis_data)
        data_layout.addLayout(data_path_layout)
        data_layout.addWidget(self.btn_export_data)
        data_group.setLayout(data_layout)

        export_layout.addWidget(result_group)
        export_layout.addWidget(data_group)
        export_group.setLayout(export_layout)
        left_layout.addWidget(export_group)

        left_layout.addStretch()
        main_layout.addWidget(left_panel, 1, 0)

        # Right image panel
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        # Image display
        self.image_label = ImageLabel()
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.image_label)
        scroll_area.setMinimumSize(600, 400)

        # Image view controls
        btn_bottom = QWidget()
        btn_layout = QHBoxLayout(btn_bottom)
        self.btn_show_original = QPushButton("Show original image")
        self.btn_show_original.clicked.connect(self.show_original_image)
        self.btn_show_mask = QPushButton("Show prediction mask")
        self.btn_show_mask.clicked.connect(self.show_masked_image)
        self.btn_show_original.setEnabled(False)
        self.btn_show_mask.setEnabled(False)
        btn_layout.addWidget(self.btn_show_original)
        btn_layout.addWidget(self.btn_show_mask)

        right_layout.addWidget(scroll_area)
        right_layout.addWidget(btn_bottom)
        main_layout.addWidget(right_panel, 1, 1)
        
        
        '''
        # Apply consistent styling
        self.setStyleSheet("""
            /* Base widget styling */
            QWidget {
                font-family: "Microsoft YaHei";
                font-size: 20px;
            }
            
            /* Buttons */
            QPushButton {
                font-size: 16px;
                min-height: 28px;
                padding: 4px 20px;
            }
            
            /* Input fields */
            QLineEdit, QTextEdit {
                font-size: 18px;
                padding: 4px;
            }
            
            /* Radio buttons and checkboxes */
            QRadioButton, QCheckBox {
                font-size: 20px;
                spacing: 5px;
            }
        
        """)
        '''
        # '''
        # Apply consistent styling
        self.setStyleSheet("""
            
            /* Buttons */
            QPushButton {
                font-size: 16px;
                min-height: 28px;
                padding: 4px 20px;
            }
            
            /* Input fields */
            QLineEdit, QTextEdit {
                font-size: 10px;
                padding: 4px;
            }
            
            /* Radio buttons and checkboxes */
            QRadioButton, QCheckBox {
                font-size: 20px;
                spacing: 5px;
            }
        
        """)
        # '''
    def init_args(self):
        class Args:
            mode = "IF"
            unet_model_path = "./configs/unet-config.yaml"
            yolo_model_path = "./weights/yolov11/yolo11m-seg.pt"
            # checkpoint = "./IF_IHC_Epithe_All_Models_Weights_Combined.pt"
            checkpoint = ""
            predict_config = "./configs/predict.yaml"
        return Args()
    
    def update_mode_and_reset_display(self, checked, mode):
        """Update mode and reset display"""
        if checked:  # Only execute when radio button is selected
            self.update_mode(checked)  # Call original update mode method
            self.selected_image_display.setText("NO Image Selected")  # Reset display
            self.current_mode = mode  # Save current mode
            
    def select_image(self, mode):
        self.image_path = None
        path, _ = QFileDialog.getOpenFileName(self, f"Select {mode} image", "", "Images (*.png *.jpg *.tif)")
        if path:
            self.image_path = path
            print(f"{mode} image selected:", path)
            new_text = f"{mode} image selected:\n{os.path.basename(path)}" # 20250610, JC, only show basename
            self.selected_image_display.setText(new_text)

    def upload_checkpoint(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select checkpoint file")
        if path:
            self.args.checkpoint = path
            self.load_models()
            
    
    def load_models(self):
        """Load models and validate"""
        if not hasattr(self.args, 'checkpoint') or not self.args.checkpoint:
            QMessageBox.warning(self, "Error", "Please upload model weights first")
            return False
        try:
            # Verify checkpoint file exists
            if not os.path.exists(self.args.checkpoint):
                QMessageBox.critical(self, "Error", "Checkpoint file not found!")
                return False
            if self.models_loaded and self.current_mode == self.args.mode:
                print(f"Models already loaded for {self.args.mode} mode")
                return True
            if self.args.mode =="Epithelial":
                model_mode="EL"
            else:
                model_mode=self.args.mode
            self.yolo, self.unet = initialize_models_all_3(
                yolo_weights_path=self.args.yolo_model_path,
                unet_model_path=self.args.unet_model_path,
                checkpoint=torch.load(self.args.checkpoint, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')), #20250610, JC, for CPU
                mode=model_mode
            )
            # Verify models loaded successfully
            if self.yolo is None or self.unet is None:
                raise RuntimeError("Failed to initialize models")
            
            self.config, _ = init_params(self.args.predict_config, self.args.mode)        
            # Update loaded status
            self.models_loaded = True
            self.current_mode = self.args.mode
            print(f"Models loaded for mode: {self.current_mode}")  # Debug info
            return True
        except Exception as e:
            self.models_loaded = False
            self.current_mode = None
            QMessageBox.critical(self, "Error", f"Model loading failed: {str(e)}")
            return False
        
    def update_mode(self, checked):
        """Update mode and reset image selection display"""
        if checked:  # Only execute when radio button is selected
            selected = self.sender().text()  # Get text of radio button that triggered event
            self.args.mode = selected
            self.selected_image_display.setText("NO Image Selected")  # Reset display
            # Reset loaded status and reload models when mode changes
            self.models_loaded = False
            self.current_mode = None
            self.load_models()  # Keep original call

            self.update_slider_defaults()

    def update_slider_defaults(self):
        """Update slider defaults based on current config"""
        if not hasattr(self, 'config'):
            return
            
        try:
            # Update slider values from config
            if 'img_pre_process' in self.config and 'overlap' in self.config['img_pre_process']:
                overlap_value = int(self.config['img_pre_process']['overlap'] * 100)
                self.sliders["overlap"].setValue(overlap_value)
                
            if 'unet_params' in self.config and 'unet_model_threshold' in self.config['unet_params']:
                seg_value = int(self.config['unet_params']['unet_model_threshold'] * 100)
                self.sliders["segmentation"].setValue(seg_value)
                
            if 'yolo_args' in self.config and 'conf' in self.config['yolo_args']:
                det_value = int(self.config['yolo_args']['conf'] * 100)
                self.sliders["detection"].setValue(det_value)
                
        except Exception as e:
            print(f"Error updating slider defaults: {str(e)}")
            # Optionally show error message
            QMessageBox.warning(self, "Warning", f"Could not update slider defaults: {str(e)}")




    def upload_image(self):
        if not hasattr(self, 'image_path') or not self.image_path:
            QMessageBox.warning(self, "Error", "Please select an image first")
            return
        try:
            path = self.image_path
            if path:
                if self.args.mode == "IF":
                    original_image = tifffile.imread(self.image_path)
                    target_channel = original_image[-1]
                    img_uint8 = cv2.normalize(target_channel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    img_rgb = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)
                    whole_image = Image.fromarray(img_rgb)
                elif self.args.mode in ['Epithelial', 'IHC','EL']:
                    whole_image = get_image(self.image_path, "IHC")

                self.original_image = copy.deepcopy(whole_image)
                self.dis_original_image = copy.deepcopy(whole_image)

                # Modify image display method
                qimg = self.pil2pixmap(whole_image)
                self.image_label.setPixmapOriginal(qimg)
                self.image_label.reset_view()
                
                # Enable display buttons
                self.btn_show_original.setEnabled(True)
                self.btn_show_mask.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Image processing failed: {str(e)}")
            # Reset state
            self.btn_show_original.setEnabled(False)
            self.btn_show_mask.setEnabled(False)
    
    def display_image(self, pil_img):
        qimg = self.pil2pixmap(pil_img)
        self.image_label.setPixmapOriginal(qimg)    

    def pil2pixmap(self, img):
        img = img.convert("RGBA" if img.mode == "RGBA" else "RGB")
        arr = np.asarray(img)

        if img.mode == "RGBA":
            format = QImage.Format_RGBA8888
            bytes_per_line = 4 * arr.shape[1]
        else:
            format = QImage.Format_RGB888
            bytes_per_line = 3 * arr.shape[1]

        if not arr.flags['C_CONTIGUOUS']:
            arr = np.ascontiguousarray(arr)

        qimg = QImage(arr.data, arr.shape[1], arr.shape[0],
                      bytes_per_line, format)
        return QPixmap.fromImage(qimg)
    
    def reset_image_view(self):
        self.image_label.reset_view()

    def reset_sliders(self):
        """Reset all sliders to default values"""
        try:
            for key, slider in self.sliders.items():
                config = next(c for c in self.slider_info.values() if c['key'] == key)
                slider.setValue(config['default'])

            self.label_rois.setText("Waiting Start: 0 ROIs")
            # Set font
            font = QFont("Microsoft YaHei", 10)
            self.label_rois.setFont(font)

        # 3. Reset image display (only when original_image exists)
            if hasattr(self, 'original_image') and self.original_image is not None:
                qimg = self.pil2pixmap(self.original_image)
                self.image_label.setPixmapOriginal(qimg)
            else:
                # Clear image display
                self.image_label.clear()
                self.image_label.setText("No image loaded")
        except Exception as e:
            print(f"Error in reset_sliders: {str(e)}")
            # Optionally show error message
            QMessageBox.warning(self, "Error", f"Failed to reset sliders: {str(e)}")

    def stop_prediction(self):
        """Stop current prediction task"""
        if self.is_predicting:
            self.should_stop = True
            self.update_status.emit("Stopping prediction...")
            self.btn_stop.setEnabled(False)

    def predict(self):
        """Start prediction task"""
        if self.is_predicting:
            return

        # Check image    
        if not hasattr(self, 'image_path') or not self.image_path:
            QMessageBox.warning(self, "Error", "Please upload an image first")
            return
        # 2. Check if models need loading (core modification)
        need_load = False
        if not self.models_loaded:  # Models never loaded
            need_load = True
        elif self.current_mode != self.args.mode:  # Mode changed, need reload
            print(f"Mode changed: {self.current_mode} -> {self.args.mode}")
            need_load = True
        
        if need_load:
            if not self.load_models():  # Try to load models
                return  # Return directly if loading fails
        
        # Check models
        if not hasattr(self, 'yolo') or not hasattr(self, 'unet'):
            if not self.load_models():  # Try to load models
                return
                
        # Verify image file exists
        if not os.path.exists(self.image_path):
            QMessageBox.critical(self, "Error", "Image file not found!")
            return

        # Set prediction state
        self.is_predicting = True
        self.should_stop = False
        self.btn_predict.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.update_status.emit("Predicting...")
        
        # Start prediction thread
        self.predict_thread = threading.Thread(
            target=self.run_prediction,
            daemon=True
        )
        self.predict_thread.start()

    def run_prediction(self):
        """Prediction logic running in background thread"""
        try:
            # Get slider parameters
            over_lap = self.sliders["overlap"].value() / 100
            unet_thresh = self.sliders["segmentation"].value() / 100
            yolo_thresh = self.sliders["detection"].value() / 100

            # Update configuration
            updated_params = {
                'img_pre_process': {"overlap": over_lap},
                "unet_params": {"unet_model_threshold": unet_thresh},
                'yolo_args': {'conf': yolo_thresh}
            }
            self.config, _ = updated_parameters(self.config, updated_params)

            # Check if should stop
            if self.should_stop: 
                raise RuntimeError("Prediction stopped by user")

            # Execute YOLO prediction
            mode = self.args.mode
            pil_image = self.dis_original_image
            yolo_results, padded_img = get_yolo_predict_results(pil_image, self.yolo, self.config)

            if self.should_stop: 
                raise RuntimeError("Prediction stopped by user")

            # Execute UNET prediction
            if mode == "EL" or mode == "Epithelial":
                unet_results = get_unet_masks_EL(yolo_results, self.unet, self.config)
            else:
                unet_results = get_unet_masks(yolo_results, self.unet, self.config)

            if self.should_stop: 
                raise RuntimeError("Prediction stopped by user")

            # Get final results
            predicted_mask, rois_map = get_total_results(
                pil_image, padded_image=padded_img, final_result=unet_results, config=self.config
            )

            rois_count = np.max(rois_map)

            mask_label_pil=plot_full_label_mask_pil(rois_map)
            num_labels, labels = cv2.connectedComponents(np.array(mask_label_pil))
            final_bounding_boxes=[]
            for label in range(1, num_labels):
                # Get pixels of current connected component
                points = np.column_stack(np.where(labels == label))
                
                # Calculate bounding box (x, y, w, h)
                x, y, w, h = cv2.boundingRect(points)
                
                final_bounding_boxes.append((x, y, w, h,len(points)))
            columns = ['x', 'y', 'w', 'h', 'num_points']
            # Create DataFrame
            self.analysis_data = pd.DataFrame(final_bounding_boxes, columns=columns)

            if self.should_stop: 
                raise RuntimeError("Prediction stopped by user")

            # Generate mask image
            if mode in ["EL", "IHC","Epithelial"]:
                if mode=="Epithelial":
                    u_mode="EL"
                else:
                    u_mode =  mode
                result, _ = generate_mask_EL_IHC(pil_image, predicted_mask, u_mode)
            else:
                result, _ = generate_mask_IF(pil_image, predicted_mask)

            # Update results
            self.masked_image = result
            self.show_mask = True
            
            # Send completion signal
            self.prediction_finished.emit(rois_count)

        except Exception as e:
            # Send failure signal
            self.prediction_failed.emit(str(e))

    def on_prediction_finished(self, rois_count):
        """Handle when prediction completes successfully"""
        self.is_predicting = False
        self.btn_predict.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.update_status.emit(f"Prediction finished: {rois_count} ROIs")
        self.update_image()
        self.btn_show_mask.setEnabled(True)
        self.btn_show_original.setEnabled(True)

    def on_prediction_failed(self, error_msg):
        """Handle when prediction fails"""
        self.is_predicting = False
        self.btn_predict.setEnabled(True)
        self.btn_stop.setEnabled(False)
        
        if "stopped by user" in error_msg:
            self.update_status.emit("Prediction stopped")
        else:
            self.update_status.emit("Prediction failed")
            QMessageBox.critical(self, "Error", f"Prediction error: {error_msg}")
    
    def browse_export_path(self, export_type):
        """Select export path"""
        path = QFileDialog.getExistingDirectory(self, f"Select {export_type} export directory")
        if path:
            if export_type == "result":
                self.line_result_path.setText(path)
            else:
                self.line_data_path.setText(path)

    def export_prediction_results(self):
        """Export prediction results"""
        if not hasattr(self, 'masked_image'):
            QMessageBox.warning(self, "Error", "No prediction results to export")
            return
        
        export_path = self.line_result_path.text()
        if not export_path:
            QMessageBox.warning(self, "Error", "Please select export directory")
            return
        
        try:
            # Add actual export prediction results code here
            # Example:
            image_name = os.path.basename(self.image_path).split(".")[0]
            self.masked_image.save(os.path.join(export_path, f"{image_name}_result.png"))
            QMessageBox.information(self, "Success", "Prediction results exported successfully")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Export failed: {str(e)}")

    def export_analysis_data(self):
        """Export analysis data"""
        if not hasattr(self, 'analysis_data'):  # Assuming analysis_data attribute exists
            QMessageBox.warning(self, "Error", "No analysis data to export")
            return
        
        export_path = self.line_data_path.text()
        if not export_path:
            QMessageBox.warning(self, "Error", "Please select export directory")
            return
        
        try:
            # Add actual export analysis data code here
            # Example: Save CSV or JSON format analysis data
            image_name = os.path.basename(self.image_path).split(".")[0]
            pd.DataFrame(self.analysis_data).to_csv(os.path.join(export_path, f"{image_name}_data.csv"))
            QMessageBox.information(self, "Success", "Analysis data exported successfully")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Export failed: {str(e)}")


    def show_original_image(self):
        self.show_mask = False
        self.update_image()

    def show_masked_image(self):
        self.show_mask = True
        self.update_image()
        
    def update_image(self):
        img = self.masked_image if self.show_mask else self.original_image
        self.display_image(img)


if __name__ == "__main__":

    

    app = QApplication(sys.argv)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())