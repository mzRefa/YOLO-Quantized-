# 🚀 Quantized YOLO Implementation

A lightweight and efficient implementation of a quantized YOLO (You Only Look Once) pipeline, designed for faster inference and reduced resource usage.

This project is structured into multiple phases to simplify setup, experimentation, and deployment.


---


## ⚙️ Features

* ⚡ **Quantized YOLO model** for faster inference.
* 🧠 **Reduced memory footprint** compared to standard models.
* 🔄 **Modular pipeline** split into clear development phases.
* 🧪 **Easy experimentation** and debugging.
* 💻 **Low-resource optimized** for systems like the i5-7200U.


---


## 📊 Why Quantization?

Quantization helps by converting 32-bit floats to 8-bit integers, which allows:

1.  **Reduced Model Size:** Saves storage and bandwidth.
2.  **Improved Inference Speed:** Faster processing on standard CPUs.
3.  **Edge Deployment:** Enables use on devices like Raspberry Pi or older laptops.


---


## 🧩 How It Works

### 🔹 Phase 1 – Setup & Preparation
The `phase1.py` script handles the initial heavy lifting:
* Environment setup and dependency checking.
* Model loading and conversion to TFLite format.
* The **Quantization process** (INT8 calibration).
* Preprocessing configuration for input layers.

<br>

### 🔹 Phase 2 – Detection Pipeline
The `phase2.py` script manages the real-time execution:
* Running the optimized YOLO model on live video.
* Performing real-time object detection (e.g., Person, Cell Phone).
* Post-processing predictions and Non-Maximum Suppression (NMS).
* Real-time visualization of bounding boxes and FPS.


---


## 🚀 Getting Started

1. **Run Phase 1** to convert and quantize the model:
   `python3 phase1.py`

2. **Run Phase 2** to start the detection feed:
   `python3 phase2.py`
