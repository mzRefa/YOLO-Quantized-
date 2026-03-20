import cv2
import numpy as np
import tensorflow as tf
import time

# --- CONFIG ---
MODEL_PATH = "yolov8n_saved_model/yolov8n_int8.tflite"
CONF_THRESHOLD = 0.30
NMS_THRESHOLD = 0.45

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

# --- INIT ---
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

cap = cv2.VideoCapture(0)
prev_time = 0

print(f"Model Ready! Input Type: {input_details[0]['dtype']}")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    orig_h, orig_w = frame.shape[:2]
    
    # 1. PRE-PROCESS (Float32 / 255.0 as required by your model)
    img = cv2.resize(frame, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_data = img.astype(np.float32) / 255.0
    input_data = np.expand_dims(input_data, axis=0)

    # 2. RUN INFERENCE
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    # 3. DECODE OUTPUT
    output = interpreter.get_tensor(output_details[0]['index'])
    
    # Handle Quantized Output if necessary
    if output.dtype == np.int8 or output.dtype == np.uint8:
        scale, zero_point = output_details[0]['quantization']
        output = (output.astype(np.float32) - zero_point) * scale

    output = np.squeeze(output)
    if output.shape[0] == 84: # Handle transpose if needed
        output = output.T

    boxes, scores, class_ids = [], [], []

    # 4. ROBUST BOX SCALING
    for row in output:
        class_scores = row[4:]
        class_id = np.argmax(class_scores)
        score = class_scores[class_id]

        if score > CONF_THRESHOLD:
            # YOLOv8 format: [center_x, center_y, width, height]
            xc, yc, nw, nh = row[:4]
            
            # DETERMINISTIC SCALING:
            if xc > 1.0:
                # Pixel-space (0-640)
                x1 = int((xc - nw/2) * (orig_w / 640))
                y1 = int((yc - nh/2) * (orig_h / 640))
                bw = int(nw * (orig_w / 640))
                bh = int(nh * (orig_h / 640))
            else:
                # Normalized-space (0.0-1.0)
                x1 = int((xc - nw/2) * orig_w)
                y1 = int((yc - nh/2) * orig_h)
                bw = int(nw * orig_w)
                bh = int(nh * orig_h)
            
            boxes.append([x1, y1, bw, bh])
            scores.append(float(score))
            class_ids.append(class_id)

    # 5. DRAW BOXES
    if len(boxes) > 0:
        indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESHOLD, NMS_THRESHOLD)
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                label = CLASSES[class_ids[i]]
                color = (0, 255, 0) if label in ['person', 'cell phone'] else (255, 0, 0)
                
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{label} {scores[i]:.2f}", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 6. UI & FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    
    max_c = np.max(output[:, 4:]) if output.size > 0 else 0
    cv2.putText(frame, f"FPS: {fps:.1f} | Max Conf: {max_c:.2f}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    cv2.imshow('YOLOv8 INT8 - Detection Fixed', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
