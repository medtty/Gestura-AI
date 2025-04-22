import gradio as gr
import tensorflow as tf
import numpy as np
import json
from PIL import Image
from fastapi import FastAPI, UploadFile, File, WebSocket, Request, Response
from fastapi.responses import StreamingResponse
import uvicorn
import cv2
import mediapipe as mp
import io
import base64
import asyncio
import time
from typing import List, Dict, Any
from pydantic import BaseModel

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
# For static images, we use static_image_mode=True
hands_static = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
# For video streams, we use static_image_mode=False for better performance
hands_video = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
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

# Model and processing parameters
MODEL_INPUT_SIZE = (224, 224)
DETECTION_FREQUENCY = 5  # Process every Nth frame for performance
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence to report a gesture

# Data models for API
class GestureResponse(BaseModel):
    class_name: str
    confidence: float
    timestamp: float
    all_predictions: Dict[str, float] = None

class StreamRequest(BaseModel):
    stream_id: str = None
    width: int = 640
    height: int = 480
    fps: int = 15

# Cache to store most recent detection results
detection_cache = {}

# Preprocess function now expects a PIL Image (already cropped)
def preprocess_image(image):
    # Ensure image is RGB before resizing and converting
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(MODEL_INPUT_SIZE)
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0).astype(np.float32)

def detect_and_crop_hand(image_rgb):
    """Detect hand in the image and return cropped hand region if found"""
    h, w = image_rgb.shape[:2]
    results = hands_static.process(image_rgb)
    
    if not results.multi_hand_landmarks:
        return None, "No hand detected"
    
    # Get the first hand detected
    hand_landmarks = results.multi_hand_landmarks[0]
    
    # Calculate bounding box from landmarks
    x_min, y_min = w, h
    x_max, y_max = 0, 0
    for landmark in hand_landmarks.landmark:
        x, y = int(landmark.x * w), int(landmark.y * h)
        if x < x_min: x_min = x
        if y < y_min: y_min = y
        if x > x_max: x_max = x
        if y > y_max: y_max = y
    
    # Add padding to the bounding box
    padding = 30
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(w, x_max + padding)
    y_max = min(h, y_max + padding)
    
    # Check for valid dimensions
    if x_min >= x_max or y_min >= y_max:
        return None, "Invalid bounding box"
    
    # Crop the hand region
    cropped_image = image_rgb[y_min:y_max, x_min:x_max]
    
    if cropped_image.size == 0:
        return None, "Empty cropped image"
    
    return cropped_image, None

def process_frame_for_gesture(frame):
    """Process a single frame for hand gesture recognition"""
    try:
        # Convert to RGB for MediaPipe
        if frame.shape[2] == 4:  # RGBA
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        elif frame.shape[2] == 3 and frame.dtype == np.uint8:
            # Assuming BGR from OpenCV
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect and crop hand
        cropped_hand, error = detect_and_crop_hand(frame)
        if error:
            return {"error": error}
        
        # Convert cropped NumPy array to PIL Image
        cropped_pil = Image.fromarray(cropped_hand)
        
        # Preprocess and predict
        processed_image = preprocess_image(cropped_pil)
        interpreter.set_tensor(input_details[0]['index'], processed_image)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        prediction = output_data[0]
        
        # Get the prediction result
        predicted_class_idx = int(np.argmax(prediction))
        confidence = float(prediction[predicted_class_idx])
        predicted_class = index_to_class.get(predicted_class_idx, f"unknown_{predicted_class_idx}")
        
        # Return prediction info
        return {
            "class": predicted_class,
            "confidence": confidence,
            "timestamp": time.time(),
            "all_predictions": {
                index_to_class.get(i, f"class_{i}"): float(prediction[i])
                for i in range(len(prediction))
            }
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

def predict(image_pil):
    """Original prediction function for Gradio interface"""
    try:
        # Convert PIL image to OpenCV format
        image_cv = np.array(image_pil)
        
        # Process the image with MediaPipe Hands
        image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
        
        # Detect hand and get cropped image
        cropped_hand, error = detect_and_crop_hand(image_rgb)
        if error:
            return {"error": error}
        
        # Convert cropped NumPy array to PIL Image
        cropped_pil = Image.fromarray(cropped_hand)
        
        # Preprocess and predict
        processed_image = preprocess_image(cropped_pil)
        interpreter.set_tensor(input_details[0]['index'], processed_image)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        prediction = output_data[0]
        
        # Get the prediction result
        predicted_class_idx = int(np.argmax(prediction))
        confidence = float(prediction[predicted_class_idx])
        predicted_class = index_to_class.get(predicted_class_idx, f"unknown_{predicted_class_idx}")
        
        return {
            "class": predicted_class,
            "confidence": confidence,
            "all_predictions": {
                index_to_class.get(i, f"class_{i}"): float(prediction[i])
                for i in range(len(prediction))
            }
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

# Define the Gradio interface
with gradio_app:
    gr.Markdown("# Hand Gesture Recognition")
    with gr.Tabs():
        with gr.TabItem("Image Upload"):
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
                examples=[["examples/two_up.jpg"], ["examples/stop.jpg"]],
                inputs=input_image
            )
        
        with gr.TabItem("Live Demo"):
            gr.Markdown("""
            ## Live Demo
            Try the live demo using your webcam!
            - Please allow camera access when prompted
            - Hold your hand gesture in front of the camera
            """)
            camera_input = gr.Image(source="webcam", streaming=True, label="Camera Input")
            live_output = gr.JSON(label="Live Detection Results")
            
            def process_camera_input(img):
                if img is None:
                    return {"message": "No image received"}
                return predict(img)
            
            camera_input.change(
                fn=process_camera_input,
                inputs=camera_input,
                outputs=live_output
            )

# Mount Gradio app to FastAPI
fastapi_app = gr.mount_gradio_app(fastapi_app, gradio_app, path="/")

# API endpoint for single image prediction
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
        return predict(image_pil)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": f"Failed to process image: {e}"}

# WebSocket endpoint for video stream processing
@fastapi_app.websocket("/api/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    try:
        # Get stream configuration
        config_data = await websocket.receive_text()
        config = json.loads(config_data)
        stream_id = config.get("stream_id", f"stream_{int(time.time())}")
        
        frame_count = 0
        last_detection_time = time.time()
        processing_interval = 1.0 / DETECTION_FREQUENCY  # Process every N frames
        
        while True:
            # Receive frame data
            data = await websocket.receive_bytes()
            
            # Decode the image
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                await websocket.send_json({"error": "Invalid frame data"})
                continue
            
            frame_count += 1
            current_time = time.time()
            
            # Process every N frames for performance
            if frame_count % DETECTION_FREQUENCY == 0 or (current_time - last_detection_time) >= processing_interval:
                # Process the frame for gesture recognition
                result = process_frame_for_gesture(frame)
                
                if "error" not in result:
                    # Cache the result
                    detection_cache[stream_id] = result
                    last_detection_time = current_time
                    # Send results back to client
                    await websocket.send_json(result)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"WebSocket error: {e}")
    finally:
        print(f"WebSocket connection closed")

# REST API endpoints for mobile integration
@fastapi_app.post("/api/video/frame")
async def process_video_frame(request: Request):
    """Process a single video frame sent from Android app"""
    try:
        # Get the raw bytes from the request
        content = await request.body()
        
        # Get stream ID from header if available
        stream_id = request.headers.get("X-Stream-ID", f"stream_{int(time.time())}")
        
        # Decode the image
        nparr = np.frombuffer(content, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return {"error": "Could not decode image data"}
        
        # Process the frame
        result = process_frame_for_gesture(frame)
        
        if "error" not in result:
            # Cache the result for this stream
            detection_cache[stream_id] = result
            # Return the result
            return result
        else:
            return result
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": f"Failed to process frame: {e}"}

@fastapi_app.get("/api/gestures")
def get_available_gestures():
    """Return all available gesture classes the model can recognize"""
    return {"gestures": list(index_to_class.values())}

@fastapi_app.get("/health")
def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy", "timestamp": time.time()}

# Documentation for Android integration
@fastapi_app.get("/")
async def root():
    return {
        "app": "Hand Gesture Recognition API",
        "usage": {
            "image_prediction": "POST /api/predict with image file",
            "video_streaming": "WebSocket /api/stream or POST frames to /api/video/frame",
            "available_gestures": "GET /api/gestures"
        },
        "android_integration": {
            "single_image": "Send image as multipart/form-data to /api/predict",
            "video_stream": "Send individual frames to /api/video/frame with X-Stream-ID header",
            "websocket": "Connect to /api/stream for bidirectional communication"
        }
    }

if __name__ == "__main__":
    # Modified for Hugging Face Spaces environment
    uvicorn.run(
        fastapi_app, 
        host="0.0.0.0", 
        port=7860,
        root_path="",
        forwarded_allow_ips="*"
    )