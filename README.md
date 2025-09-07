<div align="center">
<br>
<img src="utils/newicon.png" width="200">
<h2>CenSegNet: a generalist high-throughput deep learning framework for centrosome phenotyping at spatial and single-cell resolution in heterogeneous tissues</h2>

<br>
</div>

**CenSegNet enables the first large-scale, spatially resolved quantification of numerical and structural Centrosome Amplification at single-cell resolution**

We provide both a user-friendly Graphical User Interface (GUI) for ease of use, as well as a few command-lines for terminal-based operation. Detailed instructions for both methods are provided below.

### :computer: GUI

[Our pre-trained weight](https://drive.google.com/file/d/1UK7EaV5llvtQHAJKET__uJ0FO30vwjE0/view?usp=sharing)  is available for download.

<pre><code>
python CenSegNet_GUI.py
</code></pre>

:mag_right: **Instructions**

<img src="utils/GUI_GIF.gif" width="1000">

### :clipboard: Command

**Detect Centrosomes for IF Image**

<pre><code>
python CenSegNet.py --mode IF \ 
		--image ./PathTo/IF.tif \
		--checkpoint ./PathTo/IF_IHC_Epithe_All_Models_Weights_Combined.pt \
		--outdir ./PathToSavePredictions/
</code></pre>

**Detect Centrosomes for IHC Image**

<pre><code>
python CenSegNet.py --mode IHC \
		--image ./PathTo/IHC.tif \
		--checkpoint ./PathTo/IF_IHC_Epithe_All_Models_Weights_Combined.pt \
		--outdir ./PathToSavePredictions/
</code></pre>

**Detect Epithelial Regions for IHC Image**

<pre><code>
python CenSegNet.py --mode Epithelial \
		--image ./PathTo/IHC.tif \
		--checkpoint ./PathTo/IF_IHC_Epithe_All_Models_Weights_Combined.pt \
		--outdir ./PathToSavePredictions/
</code></pre>

### :hammer: Dependency

Our method relies on detections from YOLO. We recommend installing the Ultralytics package to ensure compatibility. Additionally, the required **yolov11m-seg.pt** model will be automatically downloaded when running the code.


~~~sh
pip install opencv_python
pip install tifffile
pip install PyQt5
pip install ultralytics
pip install Pillow
pip install skimage
pip install PyYAML
pip install Requests
pip install scikit_learn
~~~


### :sunny: Detection/Segmentation Examples from Unseen Images

<div align="center">
<br>
<img src="utils/example.png" width="1000">
<br>
</div>

