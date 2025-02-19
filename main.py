import io
import base64
import PIL.Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO


model = "CoffeeEye.pt"
model = YOLO(model)

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
    contents = await file.read()
    pil_image = PIL.Image.open(io.BytesIO(contents))
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
