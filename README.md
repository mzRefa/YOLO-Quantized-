# ⚡ YOLO-Quantized: Edge-Optimized Object Detection

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg) 
![YOLOv11](https://img.shields.io/badge/YOLO-v11-red.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-orange.svg)
![Optimization](https://img.shields.io/badge/Optimization-Quantized-green.svg)

**YOLO-Quantized** is a specialized computer vision repository focused on maximizing inference speeds for real-time object detection. By leveraging quantized weights and lightweight neural architectures, this project achieves high-FPS detection on standard CPU hardware without the need for dedicated GPUs.

---

## 🚀 Performance Breakthroughs

### 🔹 Model Quantization
This repo focuses on the transition from **FP32 (Floating Point 32)** to more efficient formats. This reduces the model size and significantly speeds up the mathematical operations (tensor multiplication) required for detection, making it ideal for mobile and IoT devices.

### 🔹 Real-Time Neural Perception
Using the latest **YOLO11 Nano** architecture, the system is tuned for ultra-low latency. It is capable of identifying 80+ object classes in milliseconds, balancing the trade-off between Mean Average Precision (mAP) and computational cost.

### 🔹 Hardware Agnostic Deployment
While most YOLO implementations "choke" on standard laptop CPUs, this version is optimized for:
* **Intel/AMD CPUs** via OpenVINO-ready logic.
* **Apple Silicon** via CoreML-compatible pathways.
* **Standard Webcams** with zero-lag stream processing.

---

## 📂 Repository Structure

```text
.
├──  Models                # Quantized .pt and .onnx weights
│   └── yolo11n.pt          # Nano-scale weights for edge speed
├──  Scripts               # Production-ready detection logic
│   └── yolo_vision.py      # Main real-time inference engine
├──  requirements.txt       # Optimized dependency list
└──  LICENSE                # MIT License
