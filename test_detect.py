from ultralytics import YOLO
from PIL import Image
import io

# Load YOLOv5 model
model = YOLO('public/best.pt')  # Ensure you have the correct path to your best.pt file

def run_inference(image_path):
    try:
        # Open image file
        image = Image.open(image_path)
        # Perform inference
        results = model(image)  # Inference on the image
        
        # Check the results for detections
        if results:
            # Render the results on the image
            results_image = results[0].plot()  # Use .plot() instead of .render()
            # Save rendered image to a byte array
            img_byte_arr = io.BytesIO()
            Image.fromarray(results_image).save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            return img_byte_arr
        else:
            print("No detections found.")
            return None

    except Exception as e:
        print(f"Error during inference: {e}")
        return None

if __name__ == "__main__":
    # Specify your input image path here
    processed_image = run_inference('public/db243_jpg.rf.6f49fa101eeda3f98be78f16037a69b5.jpg')  # Update with your actual image path
    if processed_image:
        # Save the processed image
        with open('output_image.png', 'wb') as f:
            f.write(processed_image.getvalue())
    else:
        print("Image processing failed.")