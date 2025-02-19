import os
import io
import base64
import requests
import PIL.Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

# Google Drive Model URL (Replace with your actual file ID)
MODEL_URL = "https://drive.google.com/uc?id=1ZSjvc6tbWX4rrnPjLI4HaWuez5aM7lCx"
MODEL_PATH = "CoffeeEye.pt"

def download_model():
    """Download the YOLO model if it's not available locally."""
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        try:
            response = requests.get(MODEL_URL, allow_redirects=True)
            if response.status_code == 200:
                with open(MODEL_PATH, "wb") as f:
                    f.write(response.content)
                print("Model downloaded successfully!")
            else:
                print(f"Failed to download the model. Status code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading the model: {e}")

# Ensure the model is downloaded before loading YOLO
download_model()

# Load YOLO model if available
try:
    if os.path.exists(MODEL_PATH):
        model = YOLO(MODEL_PATH)
        print("Model loaded successfully!")
    else:
        model = None
        print("Warning: Model not found! API will not work correctly.")
except Exception as e:
    model = None
    print(f"Error loading the model: {e}")

# Initialize FastAPI
app = FastAPI()

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

@app.get("/")
def home():
    """Root endpoint to check API status."""
    return {"message": "FastAPI is running with YOLO!"}

def image_to_base64(image: PIL.Image.Image) -> str:
    """Convert PIL image to base64-encoded string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

@app.post("/predict-image/")
async def predict_image(file: UploadFile = File(...)):
    """Process an uploaded image and return YOLO predictions."""
    if model is None:
        return {"error": "Model not found. Prediction is not possible."}

    try:
        # Read image and convert to PIL format
        contents = await file.read()
        pil_image = PIL.Image.open(io.BytesIO(contents))

        # Run YOLO model for predictions
        results = model.predict(pil_image)

        if results and len(results) > 0 and results[0].boxes is not None:
            result_image_array = results[0].plot()  
            result_pil_image = PIL.Image.fromarray(result_image_array)
            result_image_base64 = image_to_base64(result_pil_image)
            
            predictions = [
                f"{model.names[int(box.cls.item())]}: {box.conf.item():.2f}"
                for box in results[0].boxes
            ] if results[0].boxes else ["No predictions"]
        else:
            result_image_base64 = image_to_base64(pil_image)
            predictions = ["No predictions"]
        
        return {"prediction": predictions, "image": result_image_base64}

    except Exception as e:
        return {"error": f"An error occurred during prediction: {str(e)}"}
