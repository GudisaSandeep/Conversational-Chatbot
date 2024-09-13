import os
from dotenv import load_dotenv
import google.generativeai as genai
import google.ai.generativelanguage as glm
from PIL import Image
import gradio as gr
import numpy as np
import io
import speech_recognition as sr
from pydub import AudioSegment
import edge_tts
import asyncio
import base64
import json

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
        self.model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        self.conversation = []
        self.recognizer = sr.Recognizer()

    async def text_to_speech(self, text):
        communicate = edge_tts.Communicate(text, "en-US-AriaNeural")
        audio = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio.write(chunk["data"])
        audio.seek(0)
        return audio.getvalue()

    def process_audio(self, audio_data):
        audio = AudioSegment.from_wav(io.BytesIO(audio_data))
        audio.export("temp.wav", format="wav")
        
        with sr.AudioFile("temp.wav") as source:
            audio = self.recognizer.record(source)
        
        try:
            text = self.recognizer.recognize_google(audio)
            self.conversation.append(("You", text))

            response = self.model.generate_content(
                glm.Content(parts=[glm.Part(text=text)]),
                stream=True
            )
            response.resolve()

            self.conversation.append(("Assistant", response.text))
            return text, response.text
        except sr.UnknownValueError:
            return "Could not understand audio", "Sorry, I couldn't understand the audio."
        except sr.RequestError as e:
            return f"Error: {str(e)}", "There was an error processing your request."
        finally:
            os.remove("temp.wav")

voice_interaction = VoiceInteraction()

async def process_audio_stream(websocket):
    audio_data = b""
    async for message in websocket:
        data = json.loads(message)
        if data['type'] == 'audio':
            audio_chunk = base64.b64decode(data['data'])
            audio_data += audio_chunk
        elif data['type'] == 'end':
            user_text, assistant_response = voice_interaction.process_audio(audio_data)
            audio_response = await voice_interaction.text_to_speech(assistant_response)
            await websocket.send(json.dumps({
                'type': 'response',
                'user_text': user_text,
                'assistant_text': assistant_response,
                'audio': base64.b64encode(audio_response).decode('utf-8')
            }))
            audio_data = b""

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 AI Assistant powered by Google Gemini")
    
    # ... [Text Chat and Image Analysis tabs remain the same]

    with gr.Tab("🎙️ Live Voice Interaction"):
        audio_input = gr.Audio(source="microphone", streaming=True, type="numpy")
        start_button = gr.Button("Start Interaction")
        stop_button = gr.Button("Stop Interaction")
        user_text_output = gr.Textbox(label="You said")
        assistant_text_output = gr.Textbox(label="Assistant response")
        audio_output = gr.Audio(label="Audio response")
        conversation_output = gr.Chatbot(label="Conversation History", height=400)

        def start_interaction():
            return gr.update(visible=True), gr.update(visible=False)

        def stop_interaction():
            return gr.update(visible=False), gr.update(visible=True)

        start_button.click(start_interaction, outputs=[stop_button, start_button])
        stop_button.click(stop_interaction, outputs=[stop_button, start_button])
        
        def handle_audio(audio):
            if len(audio) == 0:
                return None, None, None, None
            audio_data = (audio * 32767).astype(np.int16).tobytes()
            user_text, assistant_response = voice_interaction.process_audio(audio_data)
            audio_response = asyncio.run(voice_interaction.text_to_speech(assistant_response))
            return user_text, assistant_response, audio_response, voice_interaction.conversation

        audio_input.stream(
            handle_audio,
            inputs=[audio_input],
            outputs=[user_text_output, assistant_text_output, audio_output, conversation_output],
            show_progress=False,
        )

demo.queue()
demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", 7860)), share=False)
