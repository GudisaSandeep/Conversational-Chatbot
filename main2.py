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
        # Convert numpy array to bytes
        pil_image = Image.fromarray(image)
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='JPEG')
        bytes_data = img_byte_arr.getvalue()
    else:
        # Assume image is already in bytes format
        bytes_data = image

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
        audio_path = "output.mp3"
        await communicate.save(audio_path)
        
        pygame.mixer.init()
        pygame.mixer.music.load(audio_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        pygame.mixer.quit()

    def listen_and_respond(self):
        with sr.Microphone() as source:
            while self.is_running:
                try:
                    #print("Listening...")
                    audio = self.recognizer.listen(source, timeout=5)
                    text = self.recognizer.recognize_google(audio)
                    #print("You said:", text)
                    self.conversation.append(("You", text))

                    response = self.model.generate_content(
                        glm.Content(parts=[glm.Part(text=text)]),
                        stream=True
                    )
                    response.resolve()

                    #print("Assistant:", response.text)
                    self.conversation.append(("Assistant", response.text))
                    asyncio.run(self.text_to_speech_and_play(response.text))
                except sr.WaitTimeoutError:
                    continue
                except sr.UnknownValueError:
                    print("Could not understand audio")
                except sr.RequestError as e:
                    print(f"Error: {str(e)}")

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

with gr.Blocks() as demo:
    gr.Markdown("# Google AI Studio + Gemini Pro")
    with gr.Tab("Text Chat"):
        text_input = gr.Textbox(label="Enter your prompt here..")
        text_button = gr.Button("Generate!")
        text_output = gr.Chatbot()
        text_button.click(text_chat, inputs=text_input, outputs=text_output)

    with gr.Tab("Image Analysis"):
        image_input = gr.Image(label="Choose an Image file")
        image_prompt = gr.Textbox(label="Enter your prompt here..")
        image_button = gr.Button("Analyze Image")
        image_output = gr.Textbox(label="Image Analysis")
        image_button.click(image_analysis, inputs=[image_input, image_prompt], outputs=image_output)

    with gr.Tab("Voice Interaction"):
        start_button = gr.Button("Start Voice Interaction")
        stop_button = gr.Button("Stop Voice Interaction")
        status_output = gr.Textbox(label="Status")
        conversation_output = gr.Chatbot(label="Conversation")
        
        start_button.click(start_voice_interaction, inputs=[], outputs=[status_output, conversation_output])
        stop_button.click(stop_voice_interaction, inputs=[], outputs=[status_output, conversation_output])
        
        gr.Markdown("The conversation will update automatically every 5 seconds.")
        demo.load(update_conversation, inputs=[], outputs=[conversation_output], every=5)
demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", 7860)), share=False)
