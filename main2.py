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
import tempfile
import speech_recognition as sr

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

class VoiceInteraction:
    def __init__(self):
        self.model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        self.conversation = []
        self.recognizer = sr.Recognizer()

    async def text_to_speech(self, text):
        communicate = edge_tts.Communicate(text, "en-US-AriaNeural")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            await communicate.save(temp_file.name)
            return temp_file.name

    def transcribe_audio(self, audio_file):
        with sr.AudioFile(audio_file) as source:
            audio = self.recognizer.record(source)
        try:
            return self.recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            return "Speech recognition could not understand the audio"
        except sr.RequestError as e:
            return f"Could not request results from speech recognition service; {e}"

    async def process_voice_input(self, audio):
        if audio is None:
            return "No audio detected. Please try again.", None, self.conversation

        user_text = self.transcribe_audio(audio)
        self.conversation.append(("You", user_text))

        response = self.model.generate_content(
            glm.Content(parts=[glm.Part(text=user_text)]),
            stream=True
        )
        response.resolve()

        self.conversation.append(("Assistant", response.text))
        
        audio_path = await self.text_to_speech(response.text)
        
        return response.text, audio_path, self.conversation

voice_interaction = VoiceInteraction()

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
        conversation_output = gr.Chatbot(label="Conversation", height=400)
        
        audio_input.change(
            lambda x: asyncio.run(voice_interaction.process_voice_input(x)), 
            inputs=[audio_input], 
            outputs=[text_output, audio_output, conversation_output]
        )

demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", 7860)), share=False)
