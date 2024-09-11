import os
from dotenv import load_dotenv
import google.generativeai as genai
import google.ai.generativelanguage as glm
from PIL import Image
import gradio as gr
import numpy as np
import io
import edge_tts
import asyncio

# Load environment variables from .env file
load_dotenv()
API_KEY = os.getenv("API_KEY")

if API_KEY is None:
    gr.Chatbot("API Key is not set. Please set the API key in the .env file.")
else:
    genai.configure(api_key=API_KEY)

def text_chat(text):
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    response = model.generate_content(
        glm.Content(
            parts=[
                glm.Part(text=text),
            ],
        ),
        stream=True
    )
    response.resolve()
    return [("You", text), ("Assistant", response.text)]

def image_analysis(image, prompt):
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image)
    elif isinstance(image, Image.Image):
        pil_image = image
    else:
        bytes_data = image

    if 'pil_image' in locals():
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='JPEG')
        bytes_data = img_byte_arr.getvalue()

    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    response = model.generate_content(
        glm.Content(
            parts=[
                glm.Part(text=prompt),
                glm.Part(inline_data=glm.Blob(mime_type='image/jpeg', data=bytes_data)),
            ],
        ),
        stream=True
    )
    response.resolve()
    return response.text

async def text_to_speech(text):
    communicate = edge_tts.Communicate(text, "en-US-AriaNeural")
    audio_path = "output.mp3"
    await communicate.save(audio_path)
    return audio_path

def voice_interaction(audio):
    if audio is None:
        return "No audio detected. Please try again.", None

    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    response = model.generate_content(
        glm.Content(
            parts=[
                glm.Part(text=audio),
            ],
        ),
        stream=True
    )
    response.resolve()
    
    # Generate speech asynchronously
    audio_path = asyncio.run(text_to_speech(response.text))
    
    return response.text, audio_path

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Interactive AI Assistance")
    gr.Markdown("# Developed by Sandeep Gudisa")
    
    with gr.Tab("💬 Text Chat"):
        with gr.Row():
            with gr.Column(scale=4):
                text_input = gr.Textbox(label="Your message", placeholder="Type your message here...")
            with gr.Column(scale=1):
                text_button = gr.Button("Send", variant="primary")
        text_output = gr.Chatbot(height=400)
        text_button.click(text_chat, inputs=text_input, outputs=text_output)

    with gr.Tab("🖼️ Image Analysis"):
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(label="Upload Image", type="pil")
            with gr.Column(scale=1):
                image_prompt = gr.Textbox(label="Prompt", placeholder="Ask about the image...")
                image_button = gr.Button("Analyze", variant="primary")
        image_output = gr.Markdown(label="Analysis Result")
        image_button.click(image_analysis, inputs=[image_input, image_prompt], outputs=image_output)

    with gr.Tab("🎙️ Voice Interaction"):
        audio_input = gr.Audio(source="microphone", type="filepath")
        text_output = gr.Textbox(label="AI Response")
        audio_output = gr.Audio(label="AI Voice Response")
        
        audio_input.change(voice_interaction, inputs=[audio_input], outputs=[text_output, audio_output])

demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", 7860)), share=False)
