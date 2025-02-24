from flask import Flask, request, jsonify
import cv2
import os
import numpy as np
import torch
from models.experimental import attempt_load
from torchvision.ops import nms
import pathlib

app = Flask(__name__)

# Load YOLOv5 model
def load_traffic_sign_model():
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
    print("Address: ", os.getcwd())
    # Load YOLOv5 model from local file
    model = attempt_load('models/best30.pt')  # Load model on CPU
    pathlib.PosixPath = temp
    return model

model = load_traffic_sign_model()
model.eval()


@app.route('/')
def home():
    return "Upload an image file for traffic sign detection"


@app.route('/detect', methods=['POST'])
def detect():
    # Receive image from request
    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    # Preprocess and perform inference
    img_tensor = preprocess_image(img)
    with torch.no_grad():
        results = model(img_tensor)[0]
    
    # Process results (same as your existing code)
    detections = process_results(results)
    
    return jsonify(detections)

def preprocess_image(img):
    img = cv2.resize(img, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)
    return img

def process_results(results):
    # Your existing code to process results
    # Define confidence & NMS thresholds
    conf_threshold = 0.25  
    nms_threshold = 0.45   

    # Reshape output (num_boxes, num_classes + 5)
    results = results.squeeze(0)  # Remove batch dimension -> (25200, num_classes + 5)

    # Extract bounding boxes, confidence scores, and class IDs correctly
    conf_scores = results[:, 4]  # Confidence score for each box
    valid_mask = conf_scores > conf_threshold  # Filter by confidence
    filtered_results = results[valid_mask]  # Keep only valid detections

    # If no detections remain, stop
    if filtered_results.shape[0] == 0:
        print("No detections found.")
        exit()

    # Extract boxes, scores, and class probabilities
    boxes = filtered_results[:, :4]  # Bounding box coordinates
    scores = filtered_results[:, 4]  # Object confidence scores
    class_probs = filtered_results[:, 5:]  # Class confidence scores

    # Get predicted class IDs
    class_ids = class_probs.argmax(dim=1)  # Get the class with highest probability

    # Convert [x, y, w, h] → [x1, y1, x2, y2]
    boxes[:, 2:] += boxes[:, :2]  # Convert width & height to absolute coords

    # Apply Non-Maximum Suppression (NMS)
    keep_indices = nms(boxes, scores, nms_threshold)
    boxes, scores, class_ids = boxes[keep_indices], scores[keep_indices], class_ids[keep_indices]

    # Process detections
    detections = []
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        conf = scores[i]
        cls = class_ids[i]

        detections.append({
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2),
            "confidence": float(conf),
            "class_id": int(cls),
            "class_name": model.names[int(cls)] if hasattr(model, 'names') else "Unknown"
        })

    return detections



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)