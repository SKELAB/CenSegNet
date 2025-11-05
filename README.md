<div align="center">
<br>
<img src="utils/newicon.png" width="200">
<h2>CenSegNet: a generalist high-throughput deep learning framework for centrosome phenotyping at spatial and single-cell resolution in heterogeneous tissues</h2>

<br>
</div>

**CenSegNet enables the first large-scale, spatially resolved quantification of numerical and structural Centrosome Amplification at single-cell resolution.**

* **Two ways to run the tool (Detailed instructions for both are provided below.):**

  * **GUI** for a quick, point-and-click workflow.
  * **CLI** for terminal/batch use.

* **Pre-trained model weights:** [Download here](https://drive.google.com/file/d/1UK7EaV5llvtQHAJKET__uJ0FO30vwjE0/view?usp=sharing).
* **Example unseen images** [for direct testing](utils/exampleTiFImages). These samples **were held out** from both training and validation and are **intentionally cropped** to a smaller resolution to meet GitHub’s single-file size limit.
* **Validated on Linux and Windows**.
### :computer: GUI

<pre><code>
python CenSegNet_GUI.py
</code></pre>

:mag_right: **Demo Usage of the GUI (Gif animation)**

<img src="utils/GUI_GIF.gif" width="1000">

### :clipboard: Command

We provide a few [example images](utils/exampleTiFImages) for direct testing of our GUI and source code.

The pre-trained weight (IF_IHC_Epithe_All_Models_Weights_Combined.pt) can be downloaded [here](https://drive.google.com/file/d/1UK7EaV5llvtQHAJKET__uJ0FO30vwjE0/view?usp=sharing).

Our method relies on detections from YOLO, the required **yolov11m-seg.pt** model will be automatically downloaded when running the code.

**Detect/segment Centrosomes for IF Image**

<pre><code>
python CenSegNet.py --mode IF \ 
		--image ./PathTo/ExampleTiFImage/IF_Centrosome.tif \
		--checkpoint ./PathTo/IF_IHC_Epithe_All_Models_Weights_Combined.pt \
		--outdir ./PathToSavePredictions/
</code></pre>

**Detect/segment Centrosomes for IHC Image**

<pre><code>
python CenSegNet.py --mode IHC \
		--image ./PathTo/ExampleTiFImage/IHC_Centrosome.tif \
		--checkpoint ./PathTo/IF_IHC_Epithe_All_Models_Weights_Combined.pt \
		--outdir ./PathToSavePredictions/
</code></pre>

**Detect/segment Epithelial Regions for IHC Image**

<pre><code>
python CenSegNet.py --mode Epithelial \
		--image ./PathTo/ExampleTiFImage/IHC_Epithelial.tif \
		--checkpoint ./PathTo/IF_IHC_Epithe_All_Models_Weights_Combined.pt \
		--outdir ./PathToSavePredictions/
</code></pre>

### :hammer: Installation & Dependency

**Typically completes in a few minutes (depends on network speed and hardware).**

If you’re on an HPC cluster with modules:

```bash
module load Python/3.10.4-GCCcore-11.3.0
```

Create and activate a virtual environment:

```bash
python -m venv censeg
source censeg/bin/activate
python -m pip install --upgrade pip
```

Install dependencies:

```bash
python -m pip install sahi pandas tifffile ultralytics PyQt5 scikit-image scikit-learn
```

(Optional) Using a `requirements.txt`:

```text
sahi
pandas
tifffile
ultralytics
PyQt5
scikit-image
scikit-learn
```

```bash
python -m pip install -r requirements.txt
```
**These dependencies are intentionally minimal; installing them will automatically pull in most commonly used libraries (e.g., torch, numpy) as transitive dependencies.**

* **Reproducibility:** Two pinned requirement sets are provided, [requirements-gpu.txt](utils/requirements/requirements-gpu.txt) and [requirements-cpu.txt](utils/requirements/requirements-cpu.txt), matching configurations where we have successfully tested the software.

* **Validated environments:**
  * **GPU (Linux):** Intel® Xeon® Gold 6226R @ 2.90 GHz, NVIDIA A100 40 GB, 128 GB RAM.
  * **CPU (Windows laptop):** 11th Gen Intel® Core™ i5-1145G7 @ 2.60 GHz, 16 GB RAM.

### :sunny: Detection/Segmentation Examples from Unseen Images

<div align="center">
<br>
<img src="utils/example.png" width="1000">
<br>
</div>

