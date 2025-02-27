from flask import Flask, request, jsonify
import cv2
import numpy as np
import torch
from models.experimental import attempt_load
from torchvision.ops import nms
import pathlib
import dlib
from scipy.spatial import distance

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_half = torch.cuda.is_available()  # Use FP16 only if CUDA is available


# Load YOLOv5 model
def load_traffic_sign_model():
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
    model = attempt_load('models/best30.pt')  # Load model on gPU
    model.to(device) # Move model to device and set it to eval mode

    if use_half:  # Convert to FP16 if using GPU
        model.half()

    pathlib.PosixPath = temp
    return model

model = load_traffic_sign_model()

# Initialize face detector and landmark predictor
face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

@app.route('/')
def home():
    return "Upload an image file on /detect for traffic sign detection or use /drowsiness for drowsiness detection"

@app.route('/detect', methods=['POST'])
def detect():
    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    img_tensor = preprocess_image(img)
    with torch.no_grad():
        results = model(img_tensor)[0]
    
    detections = process_results(results)
    return jsonify(detections)

def preprocess_image(img):
    img = cv2.resize(img, (1280, 1280))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)
    return img


# Only return the highest confidence detection
# def process_results(results):
#     conf_threshold = 0.25  
#     nms_threshold = 0.4    

#     results = results.squeeze(0)
#     conf_scores = results[:, 4]
#     valid_mask = conf_scores > conf_threshold
#     filtered_results = results[valid_mask]

#     if filtered_results.shape[0] == 0:
#         return []

#     boxes = filtered_results[:, :4]
#     scores = filtered_results[:, 4]
#     class_probs = filtered_results[:, 5:]

#     class_ids = class_probs.argmax(dim=1)
#     boxes[:, 2:] += boxes[:, :2]

#     keep_indices = nms(boxes, scores, nms_threshold)
#     boxes, scores, class_ids = boxes[keep_indices], scores[keep_indices], class_ids[keep_indices]

#     if len(scores) == 0:
#         return []

#     max_conf_index = scores.argmax().item()

#     x1, y1, x2, y2 = boxes[max_conf_index]
#     conf = scores[max_conf_index].item()
#     cls = int(class_ids[max_conf_index])

#     best_detection = {
#         "x1": int(x1),
#         "y1": int(y1),
#         "x2": int(x2),
#         "y2": int(y2),
#         "confidence": conf,
#         "class_id": cls,
#         "class_name": model.names[cls] if hasattr(model, 'names') else "Unknown"
#     }

#     return [best_detection]  # Return as a list for JSON format


# Return the highest and second-highest confidence detections with different class_ids
def process_results(results):
    conf_threshold = 0.3 
    nms_threshold = 0.4    

    results = results.squeeze(0)
    conf_scores = results[:, 4]
    valid_mask = conf_scores > conf_threshold
    filtered_results = results[valid_mask]

    if filtered_results.shape[0] == 0:
        return []

    boxes = filtered_results[:, :4]
    scores = filtered_results[:, 4]
    class_probs = filtered_results[:, 5:]

    class_ids = class_probs.argmax(dim=1)
    boxes[:, 2:] += boxes[:, :2]

    keep_indices = nms(boxes, scores, nms_threshold)
    boxes, scores, class_ids = boxes[keep_indices], scores[keep_indices], class_ids[keep_indices]

    if len(scores) == 0:
        return []

    sorted_indices = scores.argsort(descending=True)
    max_conf_index = sorted_indices[0].item()  

    best_detection = {
        "x1": int(boxes[max_conf_index][0]),
        "y1": int(boxes[max_conf_index][1]),
        "x2": int(boxes[max_conf_index][2]),
        "y2": int(boxes[max_conf_index][3]),
        "confidence": scores[max_conf_index].item(),
        "class_id": int(class_ids[max_conf_index]),
        "class_name": model.names[int(class_ids[max_conf_index])] if hasattr(model, 'names') else "Unknown"
    }

    detections = [best_detection]  # Store the best detection
    best_class_id = int(class_ids[max_conf_index])

    for i in range(1, len(scores)):  # Find the second best detection with a different class_id
        second_conf_index = sorted_indices[i].item()
        second_class_id = int(class_ids[second_conf_index])
        
        if second_class_id != best_class_id and scores[second_conf_index].item() > 0.5:
            second_best_detection = {
                "x1": int(boxes[second_conf_index][0]),
                "y1": int(boxes[second_conf_index][1]),
                "x2": int(boxes[second_conf_index][2]),
                "y2": int(boxes[second_conf_index][3]),
                "confidence": scores[second_conf_index].item(),
                "class_id": second_class_id,
                "class_name": model.names[second_class_id] if hasattr(model, 'names') else "Unknown"
            }
            detections.append(second_best_detection)
            break  # Stop after finding one valid second detection

    return detections



@app.route('/drowsiness', methods=['POST'])
def drowsiness_detection():
    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray_scale)

    drowsiness_results = []
    for face in faces:
        face_landmarks = dlib_facelandmark(gray_scale, face)
        leftEye = []
        rightEye = []

        for n in range(36, 42):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            rightEye.append((x, y))

        for n in range(42, 48):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            leftEye.append((x, y))

        right_Eye = Detect_Eye(rightEye)
        left_Eye = Detect_Eye(leftEye)
        Eye_Rat = (left_Eye + right_Eye) / 2
        Eye_Rat = round(Eye_Rat, 2)

        if Eye_Rat < 0.25:
            drowsiness_results.append({
                "drowsiness_detected": True,
                "eye_aspect_ratio": Eye_Rat,
                "message": "Drowsiness Detected. Stop driving to prevent accidents."
            })
        else:
            drowsiness_results.append({
                "drowsiness_detected": False,
                "eye_aspect_ratio": Eye_Rat,
                "message": "No drowsiness detected."
            })

    return jsonify(drowsiness_results)

def Detect_Eye(eye):
    poi_A = distance.euclidean(eye[1], eye[5])
    poi_B = distance.euclidean(eye[2], eye[4])
    poi_C = distance.euclidean(eye[0], eye[3])
    aspect_ratio_Eye = (poi_A + poi_B) / (2 * poi_C)
    return aspect_ratio_Eye

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)