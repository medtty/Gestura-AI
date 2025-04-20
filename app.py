import gradio as gr
import tensorflow as tf
import numpy as np
import json
from PIL import Image
from fastapi import FastAPI, UploadFile, File
import uvicorn
import cv2 # Import OpenCV
import mediapipe as mp # Import MediaPipe

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Create both Gradio and FastAPI apps
gradio_app = gr.Blocks()
fastapi_app = FastAPI()

# Load model and class indices
interpreter = tf.lite.Interpreter(model_path="model/model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open('model/class_indices.json') as f:
    class_indices = json.load(f)

index_to_class = {int(k): v for k, v in class_indices.items()}

# Preprocess function now expects a PIL Image (already cropped)
def preprocess_image(image):
    # Ensure image is RGB before resizing and converting
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    # The input tensor is expected to be float32
    return np.expand_dims(image_array, axis=0).astype(np.float32)

# Modified predict function to include hand detection and cropping
def predict(image_pil):
    try:
        print(f"Original image mode: {image_pil.mode}, size: {image_pil.size}")

        # Convert PIL image to OpenCV format (NumPy array)
        image_cv = np.array(image_pil)
        # Convert RGB (from PIL) to BGR (for OpenCV display if needed) then back to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)

        # Process the image with MediaPipe Hands
        results = hands.process(image_rgb)

        if not results.multi_hand_landmarks:
            print("No hand detected in the image.")
            return {"error": "No hand detected"}

        # Assuming only one hand is detected (max_num_hands=1)
        hand_landmarks = results.multi_hand_landmarks[0]

        # Calculate bounding box from landmarks
        h, w, _ = image_rgb.shape
        x_min, y_min = w, h
        x_max, y_max = 0, 0
        for landmark in hand_landmarks.landmark:
            x, y = int(landmark.x * w), int(landmark.y * h)
            if x < x_min: x_min = x
            if y < y_min: y_min = y
            if x > x_max: x_max = x
            if y > y_max: y_max = y

        # Add some padding to the bounding box
        padding = 30
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)

        # Ensure the box has valid dimensions
        if x_min >= x_max or y_min >= y_max:
             print("Invalid bounding box calculated.")
             return {"error": "Could not calculate valid hand bounding box"}

        # Crop the original RGB image using the bounding box
        cropped_image_np = image_rgb[y_min:y_max, x_min:x_max]

        # Check if cropping resulted in an empty image
        if cropped_image_np.size == 0:
            print("Cropping resulted in an empty image.")
            return {"error": "Cropping failed, possibly invalid bounding box"}

        # Convert cropped NumPy array back to PIL Image
        cropped_image_pil = Image.fromarray(cropped_image_np)
        print(f"Cropped image size: {cropped_image_pil.size}")

        # Preprocess the cropped image
        processed_image = preprocess_image(cropped_image_pil)
        print(f"Processed image shape: {processed_image.shape}, dtype: {processed_image.dtype}")

        # --- Inference ---
        interpreter.set_tensor(input_details[0]['index'], processed_image)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        prediction = output_data[0]
        # --- End Inference ---

        print(f"Raw prediction output: {prediction}")

        predicted_class_idx = int(np.argmax(prediction))
        confidence = float(prediction[predicted_class_idx])
        # Use the correct class mapping loaded earlier
        predicted_class = index_to_class.get(predicted_class_idx, f"unknown_{predicted_class_idx}")

        print(f"Predicted class index: {predicted_class_idx}, Confidence: {confidence}, Class: {predicted_class}")

        return {
            "class": predicted_class,
            "confidence": confidence,
            "all_predictions": {
                # Use the correct class mapping here too
                index_to_class.get(i, f"class_{i}"): float(prediction[i])
                for i in range(len(prediction))
            }
        }
    except Exception as e:
        print(f"Error during prediction: {e}")
        # Also print traceback for detailed debugging
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

# Gradio Interface
with gradio_app:
    gr.Markdown("# Hand Gesture Recognition")
    with gr.Row():
        input_image = gr.Image(type="pil", label="Upload Image")
        output_json = gr.JSON(label="Prediction Results")
    submit = gr.Button("Predict")
    submit.click(
        fn=predict,
        inputs=input_image,
        outputs=output_json
    )
    gr.Examples(
        examples=[["examples/two_up.jpg"], ["examples/call.jpg"], ["examples/stop.jpg"]],
        inputs=input_image
    )

# Mount Gradio app to FastAPI
fastapi_app = gr.mount_gradio_app(fastapi_app, gradio_app, path="/")

# API endpoint
@fastapi_app.post("/api/predict")
async def api_predict(file: UploadFile = File(...)):
    try:
        # Read image bytes
        contents = await file.read()
        # Decode image using OpenCV
        nparr = np.frombuffer(contents, np.uint8)
        img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_cv is None:
            return {"error": "Could not decode image"}
        # Convert BGR (OpenCV default) to RGB for PIL
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(img_rgb)
        return predict(image_pil) # Call the modified predict function
    except Exception as e:
        print(f"Error processing uploaded file: {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"Failed to process image: {e}"}


if __name__ == "__main__":
    # Modified for Hugging Face Spaces environment
    uvicorn.run(
        fastapi_app, 
        host="0.0.0.0", 
        port=7860,
        root_path="",
        forwarded_allow_ips="*"
    )