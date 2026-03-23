from ultralytics import YOLO

def export_v8_quantized():
    
    model = YOLO("models/yolov8n.pt") 

    print("🚀 Quantizing YOLOv8n to INT8 ONNX...")
    
   
    model.export(format="onnx", int8=True)
    
    print("✅ Done! Check your 'models' folder for the .onnx file.")

if __name__ == "__main__":
    export_v8_quantized()
