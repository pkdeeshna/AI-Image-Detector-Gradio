!pip install transformers torch pillow gradio

from transformers import pipeline
import gradio as gr
from PIL import Image
import requests
from io import BytesIO

# Load free model (runs on Colab GPU)
pipe = pipeline("image-classification", model="umm-maybe/AI-image-detector")

def detect_ai(image):
    result = pipe(image)
    label = result[0]['label']  # 'AI Generated Image' or 'Real Image'
    confidence = result[0]['score'] * 100
    return f"{label}: {confidence:.1f}% confident"

# Simple UI for upload
demo = gr.Interface(fn=detect_ai, inputs=gr.Image(type="pil"), outputs="text")
demo.launch(share=True) 
