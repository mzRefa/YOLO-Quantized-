import cv2
import numpy as np
import tensorflow as tf
import time

# --- CONFIGURATION ---
MODEL_PATH = "yolov8n_saved_model/yolov8n_int8.tflite"
CONF_THRESHOLD = 0.28  # Lowered to help detect the phone more consistently
NMS_THRESHOLD = 0.45   # Standard overlap filtering

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH, num_threads=2)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

cap = cv2.VideoCapture(0)
prev_time = 0
frame_count = 0
active_detections = [] # Stores boxes between AI skips

print("🚀 Starting Balanced Inference. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    frame_count += 1
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    # --- AI STEP (Every 2nd Frame) ---
    if frame_count % 2 == 0:
        # 1. Pre-process (640x640 Float32)
        img = cv2.resize(frame, (640, 640))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_data = (img.astype(np.float32) / 255.0)[np.newaxis, :]

        # 2. Run Inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        # 3. Process Output [1, 84, 8400] -> [8400, 84]
        output = np.squeeze(interpreter.get_tensor(output_details[0]['index'])).T 
        
        # 4. Filter by Confidence (Vectorized for speed)
        scores = np.max(output[:, 4:], axis=1)
        mask = scores > CONF_THRESHOLD
        valid_output = output[mask]
        valid_scores = scores[mask]

        candidate_boxes = []
        candidate_ids = []
        candidate_scores = []

        for i in range(len(valid_output)):
            row = valid_output[i]
            xc, yc, nw, nh = row[:4]
            class_id = np.argmax(row[4:])
            
            # Scale coordinates to frame size
            bx = int((xc - nw/2) * w)
            by = int((yc - nh/2) * h)
            bw = int(nw * w)
            bh = int(nh * h)
            
            candidate_boxes.append([bx, by, bw, bh])
            candidate_ids.append(class_id)
            candidate_scores.append(float(valid_scores[i]))

        # 5. Non-Maximum Suppression (Fixes the multiple box issue)
        active_detections = []
        if len(candidate_boxes) > 0:
            indices = cv2.dnn.NMSBoxes(candidate_boxes, candidate_scores, CONF_THRESHOLD, NMS_THRESHOLD)
            if len(indices) > 0:
                for i in indices.flatten():
                    active_detections.append({
                        "box": candidate_boxes[i],
                        "class": candidate_ids[i],
                        "score": candidate_scores[i]
                    })

    # --- RENDERING STEP (Every Frame) ---
    for det in active_detections:
        bx, by, bw, bh = det["box"]
        class_name = CLASSES[det['class']]
        score = det['score']
        
        # Color: Green for Person/Phone, Blue for others
        color = (0, 255, 0) if class_name in ['person', 'cell phone'] else (255, 100, 0)
        
        cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), color, 2)
        cv2.putText(frame, f"{class_name} {score:.2f}", (bx, by-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # FPS Calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('Quantized YOLOv8 - Optimized Feed', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
