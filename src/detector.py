from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_path):
        # Load the model (supports .pt or .onnx for quantization)
        self.model = YOLO(model_path)

    def predict_and_draw(self, frame):
        # Perform inference
        results = self.model(frame, conf=0.5, verbose=False)
        
        # Plot results on the frame
        annotated_frame = results[0].plot()
        return annotated_frame
