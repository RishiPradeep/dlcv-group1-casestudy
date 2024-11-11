# python -m uvicorn server:app --reload
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import numpy as np
import torch
from torchvision import models, transforms
import random
import cv2

from ultralytics import YOLO
from transformers import AutoModelForObjectDetection, AutoFeatureExtractor

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLO model once during startup
yolo_model = YOLO('public/best.pt')

# Load Mask R-CNN model once during startup
maskrcnn_model = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
maskrcnn_model.eval()

# Preprocessing transforms for Mask R-CNN
transform = transforms.Compose([transforms.ToTensor()])

# Load YOLOS model and feature extractor once during startup
yolos_feature_extractor = AutoFeatureExtractor.from_pretrained("C:/Users/rishi/OneDrive/Desktop/Stuff/Projects/my-yolov8-app/yolo_model_directory")
yolos_model = AutoModelForObjectDetection.from_pretrained("C:/Users/rishi/OneDrive/Desktop/Stuff/Projects/my-yolov8-app/yolo_model_directory")
yolos_model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
yolos_model.to(device)

@app.post("/inference/yolo/")
async def run_yolo_inference(file: UploadFile = File(...)):
    try:
        # Read uploaded image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))

        # Perform YOLO inference
        results = yolo_model(image)
        
        if results:
            results_image = results[0].plot()  # Render detections

            # Convert to byte stream
            img_byte_arr = io.BytesIO()
            Image.fromarray(results_image).save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)

            # Return image
            return StreamingResponse(img_byte_arr, media_type="image/png")
        else:
            return {"message": "No detections found."}
    
    except Exception as e:
        return {"error": str(e)}

@app.post("/inference/maskrcnn/")
async def run_maskrcnn_inference(file: UploadFile = File(...)):
    try:
        # Read uploaded image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)

        # Perform Mask R-CNN inference
        with torch.no_grad():
            predictions = maskrcnn_model(image_tensor)

        # Set a threshold to filter weak detections
        score_threshold = 0.5
        image_np = np.array(image)

        # Draw masks and bounding boxes on the image
        for i in range(len(predictions[0]["scores"])):
            score = predictions[0]["scores"][i].item()
            if score >= score_threshold:
                # Get the mask, bounding box, and label
                mask = predictions[0]["masks"][i, 0].cpu().numpy()
                bbox = predictions[0]["boxes"][i].cpu().numpy().astype(int)

                # Generate a random color for each instance
                color = [random.randint(0, 255) for _ in range(3)]
                # Apply the mask to the image
                masked_image = np.zeros_like(image_np, dtype=np.uint8)
                masked_image[mask > 0.5] = color
                image_np[mask > 0.5] = image_np[mask > 0.5] * 0.5 + masked_image[mask > 0.5] * 0.5

                # Draw bounding box on the image
                image_np = cv2.rectangle(image_np, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

        # Convert the processed image to byte stream
        img_byte_arr = io.BytesIO()
        Image.fromarray(image_np).save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        # Return the image as a response
        return StreamingResponse(img_byte_arr, media_type="image/png")

    except Exception as e:
        return {"error": str(e)}

@app.post("/inference/yolos/")
async def run_yolos_inference(file: UploadFile = File(...)):
    try:
        # Read uploaded image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Preprocess image for YOLOS
        inputs = yolos_feature_extractor(images=image, return_tensors="pt").to(device)

        # Perform YOLOS inference
        with torch.no_grad():
            outputs = yolos_model(**inputs)

        # Process output (filtering based on confidence threshold)
        CONFIDENCE_THRESHOLD = 0.5
        probas = outputs.logits.softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > CONFIDENCE_THRESHOLD
        boxes = outputs.pred_boxes[0, keep].cpu()

        # Convert predicted boxes to original image size
        def rescale_bboxes(out_bbox, size):
            img_w, img_h = size
            b = [(x[0] - 0.5 * x[2], x[1] - 0.5 * x[3],
                  x[0] + 0.5 * x[2], x[1] + 0.5 * x[3]) for x in out_bbox]
            return torch.tensor(b) * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)

        bboxes_scaled = rescale_bboxes(boxes, image.size)

        # Draw boxes on the image
        draw = Image.fromarray(np.array(image))
        for box in bboxes_scaled.tolist():
            draw = cv2.rectangle(np.array(draw), (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)

        # Convert to byte stream
        img_byte_arr = io.BytesIO()
        Image.fromarray(draw).save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        return StreamingResponse(img_byte_arr, media_type="image/png")

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
