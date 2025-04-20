import gradio as gr
import tensorflow as tf
import numpy as np
import json
from PIL import Image
from fastapi import FastAPI, UploadFile, File
import uvicorn

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

def preprocess_image(image):
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0).astype(np.float32)

def predict(image):
    try:
        processed_image = preprocess_image(image)
        interpreter.set_tensor(input_details[0]['index'], processed_image)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        prediction = output_data[0]
        
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
    image = Image.open(file.file)
    return predict(image)

if __name__ == "__main__":
    # Modified for Hugging Face Spaces environment
    uvicorn.run(
        fastapi_app, 
        host="0.0.0.0", 
        port=7860,
        root_path="",
        forwarded_allow_ips="*"
    )