import streamlit as st
import base64
import requests
import os
from PIL import Image
import io
import time
import speech_recognition as sr
import pyttsx3
import tempfile
import yaml
from together import Together
import threading
from dotenv import load_dotenv

class VisionDescriber:
    def __init__(self, config_path="Configs/config.yml"):
        load_dotenv()
        self.config = self._load_config(config_path)
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key or api_key == "YOUR_TOGETHER_API_KEY_HERE":
            st.error("`TOGETHER_API_KEY` not found or not set. Please create a `.env` file in the project root and add `TOGETHER_API_KEY='your_key_here'`.")
            st.stop()
        self.client = Together(api_key=api_key)

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _load_config(self, config_path):
        """Loads the configuration from the YAML file."""
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            print(f"Error: config.yml not found at {config_path}")
            return None
        except yaml.YAMLError as e:
            print(f"Error parsing config.yml: {e}")
            return None

    def get_description(self, imagePath, system_prompt=None):
        """
        Gets a description of an image using the Together AI Vision model.

        Args:
            image_url (str): URL of the image to describe.
            user_prompt (str, optional): Additional user prompt. Defaults to None.

        Returns:
            str: The description of the image, or None if an error occurs.
        """
        if self.config is None:
            return None

        base64_image = self.encode_image(imagePath)
        
        stream = self.client.chat.completions.create(
            model=self.config["VisionPal"]["model"],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": system_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
            stream=True,
        )
        
        description = ""
        for chunk in stream:
            chunk_text = chunk.choices[0].delta.content or "" if chunk.choices else ""
            description += chunk_text
            
        return description
    
def setup_configs():
    """Create default config and .env files if they don't exist."""
    # Create Configs directory
    os.makedirs("Configs", exist_ok=True)

    # Create config.yml
    config_path = "Configs/config.yml"
    if not os.path.exists(config_path):
        default_config = {
            "VisionPal": {
                "model": "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo"
            }
        }
        with open(config_path, "w") as f:
            yaml.dump(default_config, f, default_flow_style=False)
        st.info("Created default `Configs/config.yml` file.")

    # Create .env
    if not os.path.exists(".env"):
        with open(".env", "w") as f:
            f.write('TOGETHER_API_KEY="YOUR_TOGETHER_API_KEY_HERE"\n')
        st.info("Created default `.env` file. Please add your API key to it and refresh the page.")

st.set_page_config(page_title="Vision Pal", page_icon="\U0001F441", layout="centered")

setup_configs()

# Initialize session states
if 'language' not in st.session_state:
    st.session_state.language = 'English'
if 'image_uploaded' not in st.session_state:
    st.session_state.image_uploaded = False
if 'tts_engine' not in st.session_state:
    st.session_state.tts_engine = pyttsx3.init()
if 'playing_audio' not in st.session_state:
    st.session_state.playing_audio = False
if 'temp_image_path' not in st.session_state:
    st.session_state.temp_image_path = None
if 'image_data' not in st.session_state:
    st.session_state.image_data = None
if 'listening' not in st.session_state:
    st.session_state.listening = False
if 'response_text' not in st.session_state:
    st.session_state.response_text = ""
if 'asking_question' not in st.session_state:
    st.session_state.asking_question = False
if 'ask_method' not in st.session_state:
    st.session_state.ask_method = None
if 'followup_question' not in st.session_state:
    st.session_state.followup_question = ""
if 'show_text_input' not in st.session_state:
    st.session_state.show_text_input = False

# Initialize vision model after setup and session state
describer = VisionDescriber()

# TTS function with interrupt support
def speak(text, lang='en'):
    def _speak():
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        for voice in voices:
            if (lang == 'ar' and 'Arabic' in voice.name) or (lang == 'en' and 'English' in voice.name):
                engine.setProperty('voice', voice.id)
                break
        st.session_state.playing_audio = True
        engine.say(text)
        engine.runAndWait()
        st.session_state.playing_audio = False

    threading.Thread(target=_speak).start()

# Stop TTS
def stop_audio():
    engine = pyttsx3.init()
    engine.stop()
    st.session_state.playing_audio = False

# Recognize speech

def recognize_speech(lang='en-US'):
    st.session_state.listening = True
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎙️ Listening...")
        r.adjust_for_ambient_noise(source, duration=1)
        audio = r.listen(source)
    st.session_state.listening = False
    try:
        return r.recognize_google(audio, language=lang)
    except:
        return ""

# Title
st.markdown("""
    <h1 style='text-align: center;'>
        <span style="color: #FF5733;">V</span><span style="color: #FFBD33;">i</span><span style="color: #DBFF33;">s</span><span style="color: #75FF33;">i</span><span style="color: #33FF57;">o</span><span style="color: #33FFBD;">n</span>
        <span style="color: #33DBFF;">P</span><span style="color: #3375FF;">a</span><span style="color: #5733FF;">l</span>
        <img src="https://img.icons8.com/ios-filled/50/visible.png" width="35" style="vertical-align: middle;"/>
    </h1>
""", unsafe_allow_html=True)

# Mic animation
if st.session_state.playing_audio:
    st.markdown("<p style='text-align:center;'>🔊 Speaking...</p>", unsafe_allow_html=True)
elif st.session_state.listening:
    st.markdown("<p style='text-align:center; animation: pulse 1s infinite;'>🎤 Listening...</p>", unsafe_allow_html=True)

# Language selection
lang_option = st.selectbox("Choose Language / اختر اللغة", ["English", "Arabic"])
lang = 'ar' if lang_option == 'Arabic' else 'en'

# Image selection logic using session_state flags
if 'use_camera' not in st.session_state:
    st.session_state.use_camera = False
if 'use_gallery' not in st.session_state:
    st.session_state.use_gallery = False

col1, col2 = st.columns(2)
with col1:
    if st.button("Use Camera"):
        st.session_state.use_camera = True
        st.session_state.use_gallery = False
with col2:
    if st.button("Use Gallery"):
        st.session_state.use_gallery = True
        st.session_state.use_camera = False

image = None
if st.session_state.use_camera:
    image = st.camera_input("Take a photo")
elif st.session_state.use_gallery:
    image = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

# Process image only after it has been received
if image and not st.session_state.image_uploaded:
    st.session_state.image_data = image.getvalue()
    st.image(st.session_state.image_data, caption="Selected Image", use_container_width =True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(st.session_state.image_data)
        st.session_state.temp_image_path = tmp_file.name

    prompt = """Describe the most important aspects in the image for a visually impaired individual to help them avoid dangerous situations like crossing roads or obstacles or existing signs to take into concideration, and help them navigate independently — in no more than 30 words."""
    if lang == 'ar':
        prompt = """وصف أهم العناصر في الصورة لمساعدة شخص مكفوف في تجنب المخاطر والعوائق والمشي بأمان دون مساعدة في 30 كلمة أو أقل."""

    with st.spinner("Analyzing image, please wait..."):
        response = describer.get_description(st.session_state.temp_image_path, prompt)
        st.session_state.response_text = response
        st.success(response)
        speak(response, lang=lang)
        st.session_state.image_uploaded = True

# Show image again and description
if st.session_state.image_uploaded:
    if st.session_state.image_data:
        st.image(st.session_state.image_data, caption="Selected Image", use_container_width = True)
    st.success(st.session_state.response_text)

# Follow-up question with audio control
if st.session_state.image_uploaded:
    if st.button("Ask Another Question"):
        st.session_state.asking_question = True
        st.session_state.ask_method = None
        st.session_state.followup_question = ""
        st.session_state.show_text_input = False

# Step 2: Choose method (radio outside condition with key)
if st.session_state.asking_question:
    st.session_state.ask_method = st.radio("Ask by:", ["Microphone", "Keyboard"], key="ask_method_radio")

# Step 3: Handle input based on method
if st.session_state.asking_question:
    if st.session_state.ask_method == "Microphone" and st.session_state.followup_question == "":
        st.session_state.followup_question = recognize_speech('ar-SA' if lang == 'ar' else 'en-US')
    elif st.session_state.ask_method == "Keyboard":
        st.session_state.show_text_input = True

# Step 4: Show text input (keyboard)
if st.session_state.show_text_input:
    st.session_state.followup_question = st.text_input("Type your question", key="text_input_key")

# Step 5: Process question
if st.session_state.followup_question:
    st.write(f"You asked: {st.session_state.followup_question}")
    with st.spinner("Getting answer..."):
        followup_response = describer.get_description(
            st.session_state.temp_image_path,
            st.session_state.followup_question
        )
        st.session_state.response_text = followup_response
        st.success(followup_response)
        speak(followup_response, lang=lang)
        st.markdown("<p style='color: green;'>✅ Analyzed successfully</p>", unsafe_allow_html=True)
    st.session_state.asking_question = False
    st.session_state.show_text_input = False

# Show stop talking button
if st.session_state.playing_audio:
    if st.button("Stop Talking"):
        stop_audio()

# Reset button
st.markdown("---")
if st.button("Start Over"):
    st.session_state.image_uploaded = False
    st.session_state.temp_image_path = None
    st.session_state.response_text = ""
    st.session_state.image_data = None
    st.session_state.use_camera = False
    st.session_state.use_gallery = False
    stop_audio()
    st.session_state.listening = False
