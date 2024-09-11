import os
from dotenv import load_dotenv
import google.generativeai as genai
import google.ai.generativelanguage as glm
from PIL import Image
import gradio as gr
import numpy as np
import io
import speech_recognition as sr
import edge_tts
import asyncio
import pygame
import threading

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
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(image)
    elif isinstance(image, Image.Image):
        # Image is already a PIL Image
        pil_image = image
    else:
        # Assume image is already in bytes format
        bytes_data = image

    # Convert PIL Image to bytes if necessary
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



class VoiceInteraction:
    def __init__(self):
        self.is_running = False
        self.recognizer = sr.Recognizer()
        self.model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        self.conversation = []

    async def text_to_speech_and_play(self, text):
        communicate = edge_tts.Communicate(text, "en-US-AriaNeural")
        audio_path = "/tmp/output.mp3"
        await communicate.save(audio_path)
        
        # Instead of playing audio, just log that it would be played
        print(f"Audio would be played: {audio_path}")

    def listen_and_respond(self):
        while self.is_running:
            try:
                # Simulate listening by waiting for a short time
                asyncio.run(asyncio.sleep(5))
                
                # Simulate received text
                text = "Simulated user input"
                print("Simulated user said:", text)
                self.conversation.append(("You", text))

                response = self.model.generate_content(
                    glm.Content(parts=[glm.Part(text=text)]),
                    stream=True
                )
                response.resolve()

                print("Assistant:", response.text)
                self.conversation.append(("Assistant", response.text))
                asyncio.run(self.text_to_speech_and_play(response.text))
            except Exception as e:
                print(f"Error in voice interaction: {str(e)}")

    def start(self):
        self.is_running = True
        self.thread = threading.Thread(target=self.listen_and_respond)
        self.thread.start()

    def stop(self):
        self.is_running = False
        if hasattr(self, 'thread'):
            self.thread.join()

voice_interaction = VoiceInteraction()

def start_voice_interaction():
    voice_interaction.start()
    return "Voice interaction started. Speak now!", voice_interaction.conversation

def stop_voice_interaction():
    voice_interaction.stop()
    return "Voice interaction stopped.", voice_interaction.conversation

def update_conversation():
    return voice_interaction.conversation

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
        with gr.Row():
            start_button = gr.Button("Start Voice Interaction", variant="primary")
            stop_button = gr.Button("Stop Voice Interaction", variant="secondary")
        status_output = gr.Markdown(label="Status")
        conversation_output = gr.Chatbot(label="Conversation", height=400)
        
        start_button.click(start_voice_interaction, inputs=[], outputs=[status_output, conversation_output])
        stop_button.click(stop_voice_interaction, inputs=[], outputs=[status_output, conversation_output])
        
        gr.Markdown("The conversation will update automatically every 5 seconds.")
        demo.load(update_conversation, inputs=[], outputs=[conversation_output], every=5)

demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", 7860)), share=False)



