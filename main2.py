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
from deep_translator import GoogleTranslator
import cv2
import argparse
# Load environment variables from .env file
load_dotenv()
API_KEY = os.getenv("API_KEY")
if API_KEY is None:
    gr.Chatbot("API Key is not set. Please set the API key in the .env file.")
else:
    genai.configure(api_key=API_KEY)

# GIF paths
GIF_LISTENING = "Images/listening.gif"
GIF_THINKING = "Images/Thinking1.gif"
GIF_SPEAKING = "Images/speaking.gif"
GIF_NEUTRAL = "Images/Neutral.gif"

# Available voices and languages
VOICES = {
    "English": {"Female": "en-US-JennyNeural", "Male": "en-US-GuyNeural"},
    "Hindi": {"Female": "hi-IN-SwaraNeural", "Male": "hi-IN-MadhurNeural"},
    "Spanish": {"Female": "es-ES-ElviraNeural", "Male": "es-ES-AlvaroNeural"},
    "French": {"Female": "fr-FR-DeniseNeural", "Male": "fr-FR-HenriNeural"},
    "German": {"Female": "de-DE-KatjaNeural", "Male": "de-DE-ConradNeural"},
    "Japanese": {"Female": "ja-JP-NanamiNeural", "Male": "ja-JP-KeitaNeural"},
    "Telugu": {"Female": "te-IN-ShrutiNeural", "Male": "te-IN-MohanNeural"},
    "Mandarin Chinese": {"Female": "zh-CN-XiaoxiaoNeural", "Male": "zh-CN-YunyangNeural"},
    "Arabic": {"Female": "ar-EG-SalmaNeural", "Male": "ar-EG-HamedNeural"},
    "Russian": {"Female": "ru-RU-DariyaNeural", "Male": "ru-RU-NikitaNeural"},
    "Portuguese": {"Female": "pt-BR-FranciscaNeural", "Male": "pt-BR-AntonioNeural"},
    "Italian": {"Female": "it-IT-ElsaNeural", "Male": "it-IT-IsmaeleNeural"},
    "Korean": {"Female": "ko-KR-SunHyeNeural", "Male": "ko-KR-InJoonNeural"},
    "Dutch": {"Female": "nl-NL-ColetteNeural", "Male": "nl-NL-FennNeural"},
    "Swedish": {"Female": "sv-SE-HilleviNeural", "Male": "sv-SE-MattiasNeural"},
    "Polish": {"Female": "pl-PL-AgnieszkaNeural", "Male": "pl-PL-MarekNeural"},
    "Turkish": {"Female": "tr-TR-EmelNeural", "Male": "tr-TR-AhmetNeural"},
    "Indonesian": {"Female": "id-ID-GadisNeural", "Male": "id-ID-ArdiNeural"},
    "Thai": {"Female": "th-TH-AcharaNeural", "Male": "th-TH-PremwutNeural"},
    "Vietnamese": {"Female": "vi-VN-HoaiMyNeural", "Male": "vi-VN-NamMinhNeural"}
    # Add more languages and voices as needed
}

LANGUAGE_CODES = {
    "English": "en", "Hindi": "hi", "Spanish": "es",
    "French": "fr", "German": "de", "Japanese": "ja",
    "Telugu": "te", "Mandarin Chinese": "zh-CN", "Arabic": "ar",
    "Russian": "ru", "Portuguese": "pt-BR", "Italian": "it",
    "Korean": "ko", "Dutch": "nl", "Swedish": "sv",
    "Polish": "pl", "Turkish": "tr", "Indonesian": "id",
    "Thai": "th", "Vietnamese": "vi"
    # Add more language codes as needed
}


def text_chat(text, max_output_tokens):
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    
    response = model.generate_content(
        glm.Content(
            parts=[
                glm.Part(text=text),
            ],
        ),
        generation_config = {
            "temperature" : 0.7,
            "max_output_tokens" : max_output_tokens,
        } ,
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
            parts=[glm.Part(text=prompt), glm.Part(inline_data=glm.Blob(mime_type='image/jpeg', data=bytes_data))],
        ),
        generation_config = {
            "temperature" : 0.7,
            "max_output_tokens" : 100,
        } ,
    )
    
    response.resolve()
    return response.text

class VoiceInteraction:
    def __init__(self):
        self.is_running = False
        self.recognizer = sr.Recognizer()
        self.model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        self.conversation = []
        self.current_state = None
        self.current_image = None
        self.input_language = "English"
        self.output_language = "English"
        self.voice = VOICES["English"]["Female"]
        
        # Adjust the recognizer's settings for better sensitivity
        self.recognizer.energy_threshold = 300  # Lower energy threshold for detecting speech
        self.recognizer.dynamic_energy_threshold = True  # Dynamically adjust for ambient noise
        self.recognizer.pause_threshold = 0.5 

    async def text_to_speech_and_play(self, text):
        self.current_state = "Speaking"
        communicate = edge_tts.Communicate(text, self.voice)
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
            # Adjust for ambient noise before starting
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            
            while self.is_running:
                try:
                    self.current_state = "Listening"
                    audio = self.recognizer.listen(source, timeout=3, phrase_time_limit=10)
                    text = self.recognizer.recognize_google(audio, language=LANGUAGE_CODES[self.input_language])
                   
                    # Translate input to English if not already in English
                    if self.input_language != "English":
                        text = GoogleTranslator(source=LANGUAGE_CODES[self.input_language], target='en').translate(text)
                    
                    self.conversation.append(("You", text))

                    self.current_state = "Thinking"
                    if self.current_image:
                        response = image_analysis(self.current_image, text)
                    else:
                        response = self.model.generate_content(
                            glm.Content(parts=[glm.Part(text=text + "Give the Reponse in only pain text and without any Markdown Formatting ")]), 
                            generation_config = {
                                "temperature" : 0.7,
                                "max_output_tokens" : 500,
                            } ,
                    
                        )
                        response.resolve()
                        response = response.text

                    # Translate response if output language is not English
                    if self.output_language != "English":
                        response = GoogleTranslator(source='en', target=LANGUAGE_CODES[self.output_language]).translate(response)

                    self.conversation.append(("Assistant", response))
                    asyncio.run(self.text_to_speech_and_play(response))
                except sr.WaitTimeoutError:
                    continue
                except sr.UnknownValueError:
                    gr.Markdown("Could not understand audio")
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

    def set_current_image(self, image):
        self.current_image = image

    def set_languages_and_voice(self, input_language, output_language, voice_gender):
        self.input_language = input_language
        self.output_language = output_language
        self.voice = VOICES[output_language][voice_gender]

voice_interaction = VoiceInteraction()

def start_voice_interaction():
    voice_interaction.start()
    return "Voice interaction started. Speak now!", voice_interaction.conversation

def stop_voice_interaction():
    voice_interaction.stop()
    return "Voice interaction stopped.", voice_interaction.conversation

def update_conversation():
    return voice_interaction.conversation

def get_current_gif():
    state = voice_interaction.current_state
    if state == "Listening":
        return GIF_LISTENING
    elif state == "Thinking":
        return GIF_THINKING
    elif state == "Speaking":
        return GIF_SPEAKING
    else:
        return GIF_NEUTRAL

def set_image_for_voice(image):
    voice_interaction.set_current_image(image)
    return "Image set for voice interaction. You can now ask questions about it."

def set_languages_and_voice(input_language, output_language, voice_gender):
    voice_interaction.set_languages_and_voice(input_language, output_language, voice_gender)
    return f"Input language set to {input_language}, output language set to {output_language} with {voice_gender} voice"

with gr.Blocks(css="""
body { background-color: #0a0a0a; color: #e0e0e0; font-family: 'Segoe UI', sans-serif; }
.gr-button { background-color: #1f1f1f; color: #bb86fc; border: 1px solid #bb86fc; }
.gr-button:hover { background-color: #2b2b2b; box-shadow: 0 0 10px #bb86fc; }
#main-title { font-size: 72px; font-weight: bold; color: #bb86fc; text-align: center; margin-top: 20px; text-shadow: 0 0 10px #bb86fc; letter-spacing: 2px; }
#subtitle { font-size: 18px; color: #888; text-align: center; margin-bottom: 30px; }
#text-chat-output, #voice-interaction-output { background-color: #121212; color: white; }
#image-analysis-output { background-color: #121212; color: white; padding: 10px; border-radius: 10px; }
""") as demo:
    gr.HTML('<h1 id="main-title">seeNsay</h1>')
    gr.Markdown("## With ❤️ by Innov8Hers", elem_id="subtitle")

    with gr.Tab("💬 Text Chat"):
        with gr.Row():
            with gr.Column(scale=4):
                text_input = gr.Textbox(label="Your message", placeholder="Type your message here...")
                max_tokens_slider = gr.Slider(minimum=100, maximum=1000, value=500, step=50, label="Max Output Tokens")
            with gr.Column(scale=1):
                text_button = gr.Button("Send", variant="primary")
        text_output = gr.Chatbot(height=400, elem_id="text-chat-output")
        text_button.click(text_chat, inputs=[text_input, max_tokens_slider], outputs=text_output)

    with gr.Tab("🖼️ Image Analysis"):
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row():
                    image_input = gr.Image(label="Upload Image", type="pil")
                    webcam_button = gr.Button("Capture from Webcam")
            with gr.Column(scale=1):
                image_prompt = gr.Textbox(label="Prompt", placeholder="Ask about the image...")
                image_prompt_voice = gr.Audio(label="Prompt via Voice")
                image_button = gr.Button("Analyze", variant="primary")
        image_output = gr.Markdown(label="Analysis Result", elem_id="image-analysis-output")

    with gr.Tab("🎙️ Voice Interaction"):
        with gr.Row():
            start_button = gr.Button("Start Voice Interaction", variant="primary")
            stop_button = gr.Button("Stop Voice Interaction", variant="secondary")
        with gr.Row():
            input_language_dropdown = gr.Dropdown(choices=list(VOICES.keys()), label="Input Language", value="English")
            output_language_dropdown = gr.Dropdown(choices=list(VOICES.keys()), label="Output Language", value="English")
            voice_gender_dropdown = gr.Dropdown(choices=["Female", "Male"], label="Voice Gender", value="Female")
        set_language_button = gr.Button("Set Languages and Voice")
        sensitivity_slider = gr.Slider(minimum=100, maximum=1000, value=300, step=50, label="Microphone Sensitivity")

        with gr.Row():
            with gr.Column(scale=1):
                gif_output = gr.Image(label="Status", visible=True, elem_id="voice-interaction-gif")
                status_output = gr.Markdown(label="Status", elem_id="voice-interaction-status")
            with gr.Column(scale=1):
                conversation_output = gr.Chatbot(label="Conversation", height=400, elem_id="voice-interaction-output")

        refresh_gif_button = gr.Button("🔄 Refresh GIF Status")
        refresh_gif_button.click(fn=get_current_gif, inputs=[], outputs=gif_output)

        def update_sensitivity(value):
            voice_interaction.recognizer.energy_threshold = value
            return f"Microphone sensitivity set to {value}"

        sensitivity_slider.change(update_sensitivity, inputs=[sensitivity_slider], outputs=[status_output])

    gr.Markdown("## With ❤️ by Innov8Hers", elem_id="subtitle")
#demo.launch(share=True)
parser = argparse.ArgumentParser()
parser.add_argument("--server-name", default="0.0.0.0")
parser.add_argument("--server-port", default=int(os.environ.get("PORT", 7860)), type=int)
args = parser.parse_args()

demo.launch(server_name=args.server_name, server_port=args.server_port)