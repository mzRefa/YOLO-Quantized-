import cv2
from src.detector import YOLODetector

def launch():
  
    active_model = "models/yolov8n.pt" 
    
    detector = YOLODetector(active_model)
    cap = cv2.VideoCapture(0)
    
    print(f"🛰️ System Online. Using model: {active_model}")

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        result_frame = detector.predict_and_draw(frame)
        
        cv2.imshow("YOLOv8-Quantized", result_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    launch()
