import cv2
from src.detector import YOLODetector

def main():
    # Path to your model inside the models folder
    # If you don't have the file yet, YOLO will download it automatically
    model_path = "models/yolo11n.pt" 
    
    # Initialize our professional detector class
    detector = YOLODetector(model_path)
    
    # Start the webcam feed
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        # Run detection
        processed_frame = detector.predict_and_draw(frame)
        
        cv2.imshow("YOLO-Quantized Real-Time", processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
