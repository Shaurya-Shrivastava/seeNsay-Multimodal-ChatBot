# 🤖 seeNsay – Multimodal ChatBot

A powerful AI-powered chatbot that understands and responds to **text, voice, and images**! Built using **Gradio**, **Gemini (Google Generative AI)**, **Speech Recognition**, **Translation**, and **Text-to-Speech** to create a truly interactive multimodal experience.

---

## 🔥 Features

- 💬 **Text Chat** — Chat with the Gemini model via text.
- 🖼️ **Image Analysis** — Upload or capture an image and ask questions about it.
- 🎙️ **Voice Interaction** — Speak in multiple languages, get smart responses, and listen to replies in natural voices.
- 🌐 **Language Translation** — Input & Output both support multi-language translation.
- 🎞️ **GIF Feedback** — Real-time status using dynamic GIFs (Listening, Thinking, Speaking).

---

## 🛠️ Tech Stack

- `Gradio` – Frontend UI
- `Gemini (Google Generative AI)`
- `OpenCV` – Webcam Integration
- `SpeechRecognition`, `edge-tts` – Voice I/O
- `deep-translator` – Language Translation
- `Python` – Backend
- `Pygame` – Audio Playback
- `Hugging Face Spaces` – Deployment

---

## 🚀 Demo

**Live Demo:** [Coming Soon – Deployed on Hugging Face Spaces]

---

## 🖼️ Screenshots

| Text Chat | Image Analysis | Voice Interaction |
|----------|----------------|-------------------|
| ![Text](Images/text_tab.png) | ![Image](Images/image_tab.png) | ![Voice](Images/voice_tab.png) |

---

## 📦 Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/Shaurya-Shrivastava/seeNsay-Multimodal-ChatBot.git
cd seeNsay-Multimodal-ChatBot
2. Setup virtual environment (Optional but recommended)
bash
Copy
Edit
python -m venv chatbot-env
chatbot-env\Scripts\activate   # On Windows
3. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
4. Add your Google API key
Create a .env file in the root folder and add:

ini
Copy
Edit
API_KEY=your_gemini_api_key_here
5. Run the app
bash
Copy
Edit
python main2.py
🧠 Future Enhancements
🎯 Context memory across tabs

📱 Mobile responsive UI

🔒 User authentication for personalized interaction

🗣️ Whisper integration for better STT

🙌 Credits
Built with ❤️ by Innov8Hers Team

📄 License
This project is licensed under the MIT License