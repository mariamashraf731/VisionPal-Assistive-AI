import sys
import os
import base64
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, 
                            QVBoxLayout, QHBoxLayout, QWidget, QLabel, 
                            QTextEdit, QFileDialog, QComboBox, QMessageBox)
from PyQt5.QtGui import QFont, QPixmap, QImage
from PyQt5.QtCore import Qt, QSize, pyqtSignal, QObject, QTimer
import cv2
from gtts import gTTS
import tempfile
import pygame
import threading
import yaml
from together import Together
import speech_recognition as sr
from dotenv import load_dotenv

class VisionDescriber:
    def __init__(self, config_path="Configs/config.yml"):
        load_dotenv()
        self.config = self._load_config(config_path)
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key or api_key == "YOUR_TOGETHER_API_KEY_HERE":
            raise ValueError("TOGETHER_API_KEY not found or not set in .env file.")
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

class WorkerSignals(QObject):
    """Define signals for worker thread communication."""
    result = pyqtSignal(str)
    error = pyqtSignal(str)
    finished = pyqtSignal()
    speech_recognized = pyqtSignal(str)
    frame_ready = pyqtSignal(QImage)

class VisionPalApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VisionPal")
        self.setGeometry(100, 100, 900, 800)  # Made window taller for additional elements
        
        # Initialize the Vision Describer
        self.vision_describer = self.init_vision_describer()
        
        # Store the current image path
        self.current_image_path = None
        
        # Camera variables
        self.camera_active = False
        self.cap = None
        self.camera_timer = QTimer()
        self.camera_timer.timeout.connect(self.update_camera_frame)
        
        # Language options
        self.available_languages = {
            "English": {"code": "en", "default_prompt": """Describe the most important aspects in the image for a visually impaired individual to help them avoid dangerous situations like crossing roads or obstacles or existing signs to take into concideration, and help them navigate independently — in no more than 50 words."""},
            "Arabic": {"code": "ar", "default_prompt": """وصف أهم العناصر في الصورة لمساعدة شخص كفيف في تجنب المخاطر والعوائق والاشارات الموجودة ليأخذها بعين الاعتبار والمشي بأمان دون مساعدة في 50 كلمة أو أقل."""}
        }
        self.current_language = "English"

        # Initialize the UI
        self.init_ui()
        
        # Set up signal handler
        self.worker_signals = WorkerSignals()
        self.worker_signals.result.connect(self.update_description)
        self.worker_signals.error.connect(self.update_error)
        self.worker_signals.speech_recognized.connect(self.update_prompt)
        self.worker_signals.frame_ready.connect(self.display_camera_frame)
        
        # Initialize pygame for audio playback
        pygame.mixer.init()
        self.is_playing_audio = False  # Add this line

        # Initialize speech recognizer
        self.recognizer = sr.Recognizer()
        
    def init_vision_describer(self):
        """Initialize the Vision Describer."""
        try:
            # Create Configs directory if it doesn't exist
            os.makedirs("Configs", exist_ok=True)
            
            # Create config file if it doesn't exist
            if not os.path.exists("Configs/config.yml"):
                self.create_default_config()

            # Create .env file if it doesn't exist
            if not os.path.exists(".env"):
                with open(".env", "w") as f:
                    f.write('TOGETHER_API_KEY="YOUR_TOGETHER_API_KEY_HERE"\n')
                print("Created .env file. Please add your TOGETHER_API_KEY to it.")

            return VisionDescriber()
        except ValueError as e:
            QMessageBox.critical(self, "Configuration Error", f"{e}\nPlease create or update the .env file and restart the application.")
            return None
        except Exception as e:
            print(f"Error initializing Vision Describer: {e}")
            QMessageBox.critical(self, "Initialization Error", f"An unexpected error occurred: {e}")
            return None
        
    def create_default_config(self):
        """Create a default configuration file if it doesn't exist."""
        default_config = {
            "VisionPal": {
                "model": "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
                "temperature": 1,
                "system_prompt": {
                    "en": """
                    You are a digital assistant for visually impaired individuals. 
            
                    Your task is to provide detailed descriptions of their surroundings based on images they provide. 
            
                    Focus on elements crucial for safe navigation and understanding of the environment. This includes: identifying potential obstacles (e.g., curbs, steps, objects on the ground),  
            
                    determining the user's location (e.g., street name, landmarks), describing any relevant signage (e.g., street signs, business names), 
            
                    and explaining the meaning of traffic signals and signs (e.g., stop signs, crosswalk signals, traffic light colors). 
            
                    Provide clear, concise, and contextually relevant information to aid the user in navigating their environment effectively and safely.
                    """,
                    "ar": """
                    أنت مساعد رقمي للأشخاص ذوي الإعاقة البصرية.
                    
                    مهمتك هي تقديم أوصاف مفصلة لبيئتهم المحيطة بناءً على الصور التي يقدمونها.
                    
                    ركز على العناصر الضرورية للتنقل الآمن وفهم البيئة. وهذا يشمل: تحديد العوائق المحتملة (مثل الأرصفة والدرجات والأشياء الموجودة على الأرض)،
                    
                    تحديد موقع المستخدم (مثل اسم الشارع والمعالم)، ووصف أي لافتات ذات صلة (مثل لافتات الشوارع وأسماء الشركات)،
                    
                    وشرح معنى إشارات المرور والعلامات (مثل علامات التوقف وإشارات معابر المشاة وألوان إشارات المرور).
                    
                    قدم معلومات واضحة وموجزة وذات صلة بالسياق لمساعدة المستخدم في التنقل في بيئته بشكل فعال وآمن.
                    """
                }
            }
        }
        
        with open("Configs/config.yml", "w") as f:
            yaml.dump(default_config, f, default_flow_style=False)
        
        print("Created default config.yml file.")
        
    def init_ui(self):
        """Initialize the user interface."""
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
                
        # Title with eye icon
        title_layout = QHBoxLayout()

        eye_icon = QLabel()
        eye_pixmap = QPixmap("download.png") 
        eye_pixmap = eye_pixmap.scaled(40, 40, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        eye_icon.setPixmap(eye_pixmap)
        eye_icon.setAlignment(Qt.AlignCenter)

        title_label = QLabel("VisionPal")
        title_font = QFont("Arial", 24, QFont.Bold)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)

        title_layout.addStretch()
        title_layout.addWidget(eye_icon)
        title_layout.addWidget(title_label)
        title_layout.addStretch()

        main_layout.addLayout(title_layout)

        # Language selection
        lang_layout = QHBoxLayout()
        lang_layout.addWidget(QLabel("Language:"))
        
        self.language_combo = QComboBox()
        for lang in self.available_languages.keys():
            self.language_combo.addItem(lang)
        self.language_combo.currentTextChanged.connect(self.change_language)
        lang_layout.addWidget(self.language_combo)
        lang_layout.addStretch()
        
        main_layout.addLayout(lang_layout)
        
        # Buttons layout
        buttons_layout = QHBoxLayout()
        
        # Camera button
        self.camera_btn = QPushButton("Camera")
        self.camera_btn.setMinimumSize(QSize(150, 50))
        self.camera_btn.clicked.connect(self.toggle_camera)
        buttons_layout.addWidget(self.camera_btn)
        
        # Capture button (initially disabled)
        self.capture_btn = QPushButton("Capture")
        self.capture_btn.setMinimumSize(QSize(150, 50))
        self.capture_btn.clicked.connect(self.capture_image)
        self.capture_btn.setEnabled(False)
        buttons_layout.addWidget(self.capture_btn)
        
        # Gallery button
        self.gallery_btn = QPushButton("Gallery")
        self.gallery_btn.setMinimumSize(QSize(150, 50))
        self.gallery_btn.clicked.connect(self.open_gallery)
        buttons_layout.addWidget(self.gallery_btn)
        
        main_layout.addLayout(buttons_layout)
        
        # Image preview
        self.image_preview = QLabel("No image selected")
        self.image_preview.setAlignment(Qt.AlignCenter)
        self.image_preview.setMinimumHeight(300)
        self.image_preview.setStyleSheet("border: 1px solid #ccc")
        main_layout.addWidget(self.image_preview)
        
        # Microphone button and recognized text
        mic_section = QHBoxLayout()
        
        # Microphone button
        self.mic_btn = QPushButton("🎤 Speak Prompt")
        self.mic_btn.setMinimumSize(QSize(150, 40))
        self.mic_btn.clicked.connect(self.listen_for_prompt)
        self.mic_btn.setEnabled(False)  # Initially disabled until image is loaded
        mic_section.addWidget(self.mic_btn)
        
        # Process with custom prompt button
        self.process_btn = QPushButton("Process Image")
        self.process_btn.setMinimumSize(QSize(150, 40))
        self.process_btn.clicked.connect(self.process_with_custom_prompt)
        self.process_btn.setEnabled(False)  # Initially disabled until prompt is recorded
        mic_section.addWidget(self.process_btn)
        
        main_layout.addLayout(mic_section)
        
        # Recognized speech text
        main_layout.addWidget(QLabel("Recognized Voice Prompt:"))
        self.prompt_text = QTextEdit()
        self.prompt_text.setMaximumHeight(80)
        self.prompt_text.setPlaceholderText("Your spoken prompt will appear here...")
        main_layout.addWidget(self.prompt_text)
        
        # Description text area
        main_layout.addWidget(QLabel("Image Description:"))
        self.description_text = QTextEdit()
        self.description_text.setReadOnly(True)
        self.description_text.setPlaceholderText("Image description will appear here...")
        main_layout.addWidget(self.description_text)
        
        # Add this after the process button section
        control_buttons = QHBoxLayout()
        
        # Stop button
        self.stop_btn = QPushButton("⏹ Stop Playback")
        self.stop_btn.setMinimumSize(QSize(150, 40))
        self.stop_btn.clicked.connect(self.stop_playback)
        self.stop_btn.setEnabled(False)  # Disabled by default
        control_buttons.addWidget(self.stop_btn)
        
        # Ask another question button
        self.ask_again_btn = QPushButton("🗨 Ask Another Question")
        self.ask_again_btn.setMinimumSize(QSize(200, 40))
        self.ask_again_btn.clicked.connect(self.prepare_for_new_question)
        self.ask_again_btn.setEnabled(False)
        control_buttons.addWidget(self.ask_again_btn)
        
        main_layout.addLayout(control_buttons)
        
        # Status bar for messages
        self.statusBar().showMessage("Ready")

        main_widget.setStyleSheet("""
                QWidget {
                    background: qlineargradient(
                        spread:pad, x1:0, y1:0, x2:1, y2:1, 
                        stop:0 #a2cfe8, stop:1 #4a90e2
                    );
                    color: #000;
                }

                QLabel, QPushButton, QTextEdit, QComboBox {
                    background-color: rgba(255, 255, 255, 0.8);
                    border-radius: 8px;
                    padding: 5px;
                }

                QPushButton:hover {
                    background-color: #cce7ff;
                }

                QTextEdit, QComboBox {
                    border: 1px solid #999;
                }
            """)

    def prepare_for_new_question(self):
        """Reset UI for asking another question."""
        self.stop_playback()
        self.prompt_text.clear()
        self.process_btn.setEnabled(False)
        self.ask_again_btn.setEnabled(False)
        self.statusBar().showMessage("Ready for new question")

    def change_language(self, language):
        """Change the current language."""
        self.current_language = language
        self.statusBar().showMessage(f"Language changed to {language}")
        
        # Update prompt text placeholder
        placeholder = self.available_languages[language]["default_prompt"]
        self.prompt_text.setPlaceholderText(placeholder)
    
    def toggle_camera(self):
        """Toggle camera on/off."""
        if not self.camera_active:
            self.start_camera()
        else:
            self.stop_camera()
    
    def start_camera(self):
        """Start the camera and preview."""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.statusBar().showMessage("Error: Could not open camera.")
                return
                
            self.camera_active = True
            self.camera_btn.setText("Stop Camera")
            self.capture_btn.setEnabled(True)
            self.gallery_btn.setEnabled(False)
            self.camera_timer.start(30)  # Update every 30ms (approx 33 fps)
            self.statusBar().showMessage("Camera active. Adjust and click 'Capture' when ready.")
            
        except Exception as e:
            self.statusBar().showMessage(f"Camera error: {str(e)}")
    
    def stop_camera(self):
        """Stop the camera."""
        self.camera_timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.camera_active = False
        self.camera_btn.setText("Camera")
        self.capture_btn.setEnabled(False)
        self.gallery_btn.setEnabled(True)
        self.statusBar().showMessage("Camera stopped.")
    
    def update_camera_frame(self):
        """Update camera preview frame."""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Convert frame to QImage for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame_rgb.shape
                bytes_per_line = ch * w
                q_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.worker_signals.frame_ready.emit(q_image)
    
    def display_camera_frame(self, q_image):
        """Display camera frame in UI."""
        # Scale image to fit the label while maintaining aspect ratio
        scaled_pixmap = QPixmap.fromImage(q_image).scaled(
            self.image_preview.width(), 
            self.image_preview.height(),
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        self.image_preview.setPixmap(scaled_pixmap)
    
    def capture_image(self):
        """Capture current camera frame."""
        if not self.cap or not self.camera_active:
            self.statusBar().showMessage("Camera not active.")
            return
            
        ret, frame = self.cap.read()
        if not ret:
            self.statusBar().showMessage("Failed to capture image.")
            return
            
        # Save the captured image to a temporary file
        temp_dir = tempfile.gettempdir()
        temp_file = os.path.join(temp_dir, "visionpal_capture.jpg")
        cv2.imwrite(temp_file, frame)
        
        # Store the current image path
        self.current_image_path = temp_file
        self.prompt_text.clear()  # Clear spoken prompt box

        # Enable the microphone button
        self.mic_btn.setEnabled(True)
        
        # Process the image with default prompt
        self.process_image(temp_file)
        
        # Stop the camera
        self.stop_camera()
        
        # Display the captured image
        self.display_image(temp_file)
        
    def open_gallery(self):
        """Open file dialog to select an image from gallery."""
        self.statusBar().showMessage("Opening gallery...")
        
        file_dialog = QFileDialog()
        image_path, _ = file_dialog.getOpenFileName(
            self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if image_path:
            self.display_image(image_path)
            
            # Store the current image path
            self.current_image_path = image_path
            self.prompt_text.clear()  # Clear spoken prompt box

            # Enable the microphone button
            self.mic_btn.setEnabled(True)
            
            # Process the image with default prompt
            self.process_image(image_path)
        else:
            self.statusBar().showMessage("No image selected.")
    
    def display_image(self, image_path):
        """Display the selected image in the UI."""
        pixmap = QPixmap(image_path)
        
        # Scale pixmap to fit the label while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.image_preview.width(), 
            self.image_preview.height(),
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        
        self.image_preview.setPixmap(scaled_pixmap)
        self.statusBar().showMessage(f"Image loaded: {os.path.basename(image_path)}")
    
    def listen_for_prompt(self):
        """Listen for voice input and convert to text."""
        self.statusBar().showMessage("Listening for prompt...")
        self.mic_btn.setText("🎤 Listening...")
        self.mic_btn.setEnabled(False)
        
        # Run in a separate thread to prevent UI freezing
        threading.Thread(
            target=self._listen_thread,
            daemon=True
        ).start()
    
    def _listen_thread(self):
        """Background thread for speech recognition."""
        try:
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source)
                audio = self.recognizer.listen(source, timeout=5)
                
            # Get the language code for speech recognition
            lang_code = self.available_languages[self.current_language]["code"]
            
            # Use the appropriate language for recognition
            # Note: Google Speech Recognition supports limited languages
            # For Arabic, use "ar-AR" or "ar"
            recognize_lang = "ar-AR" if lang_code == "ar" else "en-US"
            
            # Recognize speech using Google Speech Recognition
            text = self.recognizer.recognize_google(audio, language=recognize_lang)
            self.worker_signals.speech_recognized.emit(text)
            
        except sr.WaitTimeoutError:
            self.worker_signals.error.emit("Listening timed out. Please try again.")
        except sr.UnknownValueError:
            self.worker_signals.error.emit("Could not understand audio. Please try again.")
        except sr.RequestError as e:
            self.worker_signals.error.emit(f"Speech recognition error: {str(e)}")
        except Exception as e:
            self.worker_signals.error.emit(f"Error during speech recognition: {str(e)}")
        
        # Re-enable the microphone button
        self.mic_btn.setText("🎤 Speak Prompt")
        self.mic_btn.setEnabled(True)
    
    def update_prompt(self, text):
        """Update the prompt text box with recognized speech."""
        self.prompt_text.setPlainText(text)
        self.statusBar().showMessage("Speech recognized.")
        
        # Enable the process button
        self.process_btn.setEnabled(True)
    
    def process_with_custom_prompt(self):
        """Process the current image with the custom prompt."""
        if not self.current_image_path:
            self.statusBar().showMessage("Please select an image first.")
            return
            
        custom_prompt = self.prompt_text.toPlainText().strip()
        if not custom_prompt:
            custom_prompt = self.available_languages[self.current_language]["default_prompt"]
            
        self.process_image(self.current_image_path, custom_prompt)
    
    def process_image(self, image_path, custom_prompt=None):
        """Process the image to get description and read it aloud."""
        self.statusBar().showMessage("Analyzing image...")
        self.description_text.setText("Analyzing image, please wait...")
        
        if not custom_prompt:
            custom_prompt = self.available_languages[self.current_language]["default_prompt"]
        
        # Run in a separate thread to prevent UI freezing
        threading.Thread(
            target=self._process_image_thread,
            args=(image_path, custom_prompt),
            daemon=True
        ).start()
    
    def _process_image_thread(self, image_path, custom_prompt):
        """Background thread for image processing."""
        try:
            if not self.vision_describer:
                raise Exception("Vision Describer is not initialized")
                
            # Get image description
            description = self.vision_describer.get_description(image_path, custom_prompt)
            
            if not description:
                description = "Could not generate description for this image."
            
            # Send the result to the main thread using signals
            self.worker_signals.result.emit(description)
            
        except Exception as e:
            error_msg = f"Error processing image: {str(e)}"
            self.worker_signals.error.emit(error_msg)
    
    def update_description(self, description):
        """Update UI with the image description."""
        self.description_text.setText(description)
        self.statusBar().showMessage("Analysis complete.")
        
        # Enable control buttons
        self.stop_btn.setEnabled(True)
        self.ask_again_btn.setEnabled(True)
        
        # Read description aloud
        self.speak_text(description)
        
    def update_error(self, error_msg):
        """Update UI with error message. This runs in the main thread."""
        self.description_text.setText(error_msg)
        self.statusBar().showMessage("Error during analysis.")
        self.mic_btn.setEnabled(True)
        
    def speak_text(self, text):
        """Convert text to speech and play it."""
        try:
            # Create a uniquely named temporary file
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
                temp_path = tmp_file.name
            
            # Get the language code
            lang_code = self.available_languages[self.current_language]["code"]
            
            # Generate speech
            tts = gTTS(text=text, lang=lang_code, slow=False)
            tts.save(temp_path)
            
            # Play the audio
            pygame.mixer.music.load(temp_path)
            pygame.mixer.music.play()
            
            # Update playback state
            self.is_playing_audio = True
            
            # Set up a timer to delete the file after playback completes
            QTimer.singleShot(10000, lambda: self.cleanup_after_playback(temp_path))
            
        except Exception as e:
            self.statusBar().showMessage(f"Speech error: {str(e)}")

    def cleanup_temp_file(self, file_path):
        """Clean up temporary audio file."""
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except:
            pass
    
    def stop_playback(self):
        """Stop current audio playback."""
        if self.is_playing_audio:
            pygame.mixer.music.stop()
            self.is_playing_audio = False
            self.statusBar().showMessage("Playback stopped")
            self.ask_again_btn.setEnabled(True)  # Enable ask again button
            
    def cleanup_after_playback(self, file_path):
        """Clean up after playback finishes or is stopped."""
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except:
            pass
        self.is_playing_audio = False

def create_config_directory():
    """Create the Configs directory and default config file if they don't exist."""
    # Create Configs directory if it doesn't exist
    os.makedirs("Configs", exist_ok=True)
    
    # Create config file if it doesn't exist
    if not os.path.exists("Configs/config.yml"):
        default_config = {
            "together_api_key": "YOUR_TOGETHER_API_KEY_HERE",
            "VisionPal": {
                "model": "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo"
            }
        }
        
        with open("Configs/config.yml", "w") as f:
            yaml.dump(default_config, f, default_flow_style=False)
        
        print("Created default config.yml file in Configs directory. Please edit it with your API key.")

if __name__ == "__main__":
    # Start the application
    app = QApplication(sys.argv)
    window = VisionPalApp()
    window.show()
    sys.exit(app.exec_())