import gradio as gr
import tensorflow as tf
import numpy as np
import cv2

# 1. Load the "Test Brain" you just created today!
model = tf.keras.models.load_model("D:/VS Code/Coding/Deepfake_Project/Phase2_Training/test_best_model.h5")

def analyze_image(input_img):
    # 2. Preprocess (The AI needs 224x224 and normalized pixels)
    img = cv2.resize(input_img, (224, 224))
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0) # Add batch dimension
    
    # 3. Predict
    prediction = model.predict(img)[0][0]
    
    # 4. Format Output
    if prediction > 0.5:
        return f"REAL (Confidence: {prediction*100:.2f}%)"
    else:
        return f"FAKE (Confidence: {(1-prediction)*100:.2f}%)"

# 5. Create the Web UI
demo = gr.Interface(
    fn=analyze_image, 
    inputs=gr.Image(), 
    outputs="text",
    title="ArtifactLens: Quick Forensic Test",
    description="Upload an image to see the model's prediction."
)

demo.launch()