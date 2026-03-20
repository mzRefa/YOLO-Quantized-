from ultralytics import YOLO

# Load the original PyTorch model
model = YOLO('yolov8n.pt')

# Export using Ultralytics' optimized pipeline
# 'int8=True' handles the calibration automatically using the COCO dataset
print("Starting Optimized INT8 Export...")
model.export(format='tflite', int8=True, imgsz=640)

print("\nDONE! Look for 'yolov8n_full_integer_quant.tflite' or similar in the 'yolov8n_saved_model' folder.")
