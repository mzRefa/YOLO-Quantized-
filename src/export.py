from ultralytics import YOLO

def export_quantized_model(model_path):
    """
    Converts a standard PyTorch model (.pt) into an optimized, 
    quantized ONNX format for high-speed CPU inference.
    """
    # 1. Load the standard model
    model = YOLO(model_path)

    print(f"🚀 Starting quantization for: {model_path}")

    # 2. Export to ONNX format with INT8 quantization
    # 'int8=True' is the magic part that compresses the math
    path = model.export(format="onnx", int8=True)

    print(f"✅ Export Complete! Optimized model saved at: {path}")

if __name__ == "__main__":
    # Point this to the model you have in your models/ folder
    export_quantized_model("models/yolo11n.pt")
