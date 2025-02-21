import os
import io
import base64
import requests
import PIL.Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import tempfile
import cv2
import numpy as np
from collections import Counter

MODEL_URL = "https://drive.google.com/uc?export=download&id=1rX3jxyr-URjFhOF_b7MBjmOfBrkteYsP"
MODEL_PATH = "CoffeeEye.pt"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        response = requests.get(MODEL_URL, allow_redirects=True)
        if response.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
            print("Model downloaded successfully!")
        else:
            print("Failed to download the model. Please check the link.")

download_model()

if os.path.exists(MODEL_PATH):
    model = YOLO(MODEL_PATH)
else:
    model = None
    
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

@app.get("/")
def home():
    return {"message": "FastAPI is running with YOLO!"}

def image_to_base64(image: PIL.Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

@app.post("/predict-image/")
async def predict_image(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not found. Prediction is not possible."}

    contents = await file.read()
    pil_image = PIL.Image.open(io.BytesIO(contents))

    results = model.predict(pil_image)

    price_mapping = {
        "espresso": 1.5,
        "cappucino": 1.8,
        "croissant": 1.5,
        "jus orange": 3.0,
        "fanta": 2.0,
        "coca": 2.0 , 
        "citronade": 3.0
    }

    total_price = 0.0
    best_accuracy = 0.0

    if results and len(results) > 0 and results[0].boxes is not None:
        result_image_array = results[0].plot()  
        result_pil_image = PIL.Image.fromarray(result_image_array)
        result_image_base64 = image_to_base64(result_pil_image)
        
        predictions = []
        for box in results[0].boxes:
            class_id = int(box.cls.item())
            class_name = model.names[class_id]
            confidence = box.conf.item()
            price = price_mapping.get(class_name, 0.0)
            total_price += price
            best_accuracy = max(best_accuracy, confidence)
            predictions.append(f"{class_name}: {confidence:.2f}, Price: {price:.3f} DT")

        if not predictions:
            predictions = ["No predictions"]
    else:
        result_image_base64 = image_to_base64(pil_image)
        predictions = ["No predictions"]

    return {
        "prediction": predictions,
        "image": result_image_base64,
        "total_price": f"{total_price:.3f} DT",
        "accuracy": f"{best_accuracy:.2%}"
    }

@app.post("/predict-video/")
async def predict_video(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not found. Prediction is not possible."}

    try:
        price_mapping = {
            "espresso": 1.5,
            "cappucino": 1.8,
            "croissant": 1.5,
            "jus orange": 3.0,
            "fanta": 2.0,
            "coca": 2.0 , 
            "citronade": 3.0
        }
        
        contents = await file.read()
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_file.write(contents)
        temp_file.close()

        # Open video with cv2
        cap = cv2.VideoCapture(temp_file.name)
        if not cap.isOpened():
            return {"error": "Could not open video file"}

        # Store frame predictions to find most common pattern
        frame_patterns = []
        frame_data = []  # Store frame and its predictions

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = PIL.Image.fromarray(frame_rgb)
            results = model.predict(frame_pil)

            if results and len(results) > 0 and results[0].boxes is not None:
                # Create a pattern string from detections
                detections = []
                for box in results[0].boxes:
                    class_id = int(box.cls.item())
                    class_name = model.names[class_id]
                    confidence = box.conf.item()
                    detections.append(f"{class_name}")
                
                # Sort detections to ensure consistent pattern matching
                pattern = ",".join(sorted(detections))
                frame_patterns.append(pattern)
                frame_data.append((frame_rgb, results[0], pattern))

        if not frame_patterns:
            return {"error": "No detections found in video"}

        # Find most common pattern
        most_common_pattern = Counter(frame_patterns).most_common(1)[0][0]

        # Get frame data with the most common pattern
        matching_frames = [(frame, results) for frame, results, pattern in frame_data if pattern == most_common_pattern]
        
        # Find the frame with highest confidence among matching frames
        best_frame = None
        best_results = None
        best_confidence = 0.0

        for frame, results in matching_frames:
            frame_confidence = max(box.conf.item() for box in results.boxes)
            if frame_confidence > best_confidence:
                best_confidence = frame_confidence
                best_frame = frame
                best_results = results

        # Process best frame
        predictions = []
        total_price = 0.0

        result_frame = best_results.plot()
        result_pil = PIL.Image.fromarray(result_frame)
        result_base64 = image_to_base64(result_pil)

        for box in best_results.boxes:
            class_id = int(box.cls.item())
            class_name = model.names[class_id]
            confidence = box.conf.item()
            price = price_mapping.get(class_name, 0.0)
            total_price += price
            predictions.append(f"{class_name}: {confidence:.2f}, Price: {price:.3f} DT")

        cap.release()
        os.unlink(temp_file.name)

        return {
            "prediction": predictions,
            "image": result_base64,
            "total_price": f"{total_price:.3f} DT",
            "accuracy": f"{best_confidence:.2%}",
            "pattern_frequency": Counter(frame_patterns)[most_common_pattern]
        }

    except Exception as e:
        return {"error": f"Error processing video: {str(e)}"}
