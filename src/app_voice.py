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
        self.setGeometry(100, 100, 900, 800)
        
        # Initialize the Vision Describer
        self.vision_describer = self.init_vision_describer()
        
        # Store the current image path
        self.current_image_path = None
        
        # Camera variables
        self.camera_active = False
        self.cap = None
        self.camera_timer = QTimer()
        self.camera_timer.timeout.connect(self.update_camera_frame)
        
        # Voice control state
        self.voice_state = "waiting_for_start"  # States: waiting_for_start, language_selection, input_method, capturing, processing
        self.selected_language = None
        self.selected_input_method = None
        
        # Language options
        self.available_languages = {
            "English": {"code": "en", "default_prompt": """Describe the most important aspects in the image for a visually impaired individual to help them avoid dangerous situations like crossing roads or obstacles or existing signs to take into concideration, and help them navigate independently — in no more than 50 words."""},
            "Arabic": {"code": "ar", "default_prompt": """وصف أهم العناصر في الصورة لمساعدة شخص مكفوف في تجنب المخاطر والعوائق والاشارات الموجودة ليأخذها بعين الاعتبار والمشي بأمان دون مساعدة في 50 كلمة أو أقل."""}
        }

        # Initialize the UI
        self.init_ui()
        
        # Set up signal handler
        self.worker_signals = WorkerSignals()
        self.worker_signals.result.connect(self.update_description)
        self.worker_signals.error.connect(self.update_error)
        self.worker_signals.speech_recognized.connect(self.process_voice_command)
        self.worker_signals.frame_ready.connect(self.display_camera_frame)
        
        # Initialize pygame for audio playback
        pygame.mixer.init()
        self.is_playing_audio = False
        
        # Initialize speech recognizer
        self.recognizer = sr.Recognizer()
        
        # Start continuous listening
        self.start_continuous_listening()
        
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
        
        # Voice status label
        self.voice_status = QLabel('Say "start" to begin')
        self.voice_status.setAlignment(Qt.AlignCenter)
        self.voice_status.setFont(QFont("Arial", 16))
        main_layout.addWidget(self.voice_status)
        
        # Image preview
        self.image_preview = QLabel("No image selected")
        self.image_preview.setAlignment(Qt.AlignCenter)
        self.image_preview.setMinimumHeight(400)
        self.image_preview.setStyleSheet("border: 1px solid #ccc")
        main_layout.addWidget(self.image_preview)
        
        # Description text area
        main_layout.addWidget(QLabel("Image Description:"))
        self.description_text = QTextEdit()
        self.description_text.setReadOnly(True)
        self.description_text.setPlaceholderText("Image description will appear here...")
        main_layout.addWidget(self.description_text)
        
        # Status bar for messages
        self.statusBar().showMessage("Ready - Say 'start' to begin")

        main_widget.setStyleSheet("""
                QWidget {
                    background: qlineargradient(
                        spread:pad, x1:0, y1:0, x2:1, y2:1, 
                        stop:0 #a2cfe8, stop:1 #4a90e2
                    );
                    color: #000;
                }

                QLabel, QTextEdit {
                    background-color: rgba(255, 255, 255, 0.8);
                    border-radius: 8px;
                    padding: 5px;
                }

                QTextEdit {
                    border: 1px solid #999;
                }
            """)

    def start_continuous_listening(self):
        """Start continuous listening for voice commands."""
        threading.Thread(target=self._continuous_listening_thread, daemon=True).start()
    
    def _continuous_listening_thread(self):
        """Background thread for continuous speech recognition."""
        while True:
            try:
                with sr.Microphone() as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    # Listen for audio with a timeout
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)
                    
                # Recognize speech
                text = self.recognizer.recognize_google(audio, language="en-US")
                self.worker_signals.speech_recognized.emit(text.lower())
                
            except sr.WaitTimeoutError:
                # Timeout is normal, continue listening
                continue
            except sr.UnknownValueError:
                # Could not understand audio, continue listening
                continue
            except sr.RequestError as e:
                print(f"Speech recognition error: {str(e)}")
                continue
            except Exception as e:
                print(f"Error during speech recognition: {str(e)}")
                continue
    
    def process_voice_command(self, command):
        """Process voice commands based on current state."""
        command = command.strip().lower()
        
        if self.voice_state == "waiting_for_start":
            if "start" in command:
                self.voice_state = "language_selection"
                self.voice_status.setText("Say 'English' or 'Arabic'")
                self.statusBar().showMessage("Language selection - Say 'English' or 'Arabic'")
                self.speak_text("Please say English or Arabic")
                
        elif self.voice_state == "language_selection":
            if "english" in command:
                self.selected_language = "English"
                self.voice_state = "input_method"
                self.voice_status.setText("Say 'camera' or 'gallery'")
                self.statusBar().showMessage("Input method selection - Say 'camera' or 'gallery'")
                self.speak_text("Please say camera or gallery")
            elif "arabic" in command:
                self.selected_language = "Arabic"
                self.voice_state = "input_method"
                self.voice_status.setText("قل 'كاميرا' أو 'معرض'")
                self.statusBar().showMessage("اختيار طريقة الإدخال - قل 'كاميرا' أو 'معرض'")
                self.speak_text("الكاميرا أو المعرض؟")
        
        elif self.voice_state == "input_method":
            if "camera" in command or "كاميرا" in command:
                self.selected_input_method = "camera"
                if self.selected_language == "Arabic":
                    self.voice_status.setText("تشغيل الكاميرا...")
                    self.statusBar().showMessage("تشغيل الكاميرا...")
                    self.speak_text("تشغيل الكاميرا. قل التقط عندما تكون مستعد")
                else:
                    self.voice_status.setText("Starting camera...")
                    self.statusBar().showMessage("Camera starting...")
                    self.speak_text("Starting camera. Say capture when ready")
                self.start_camera_for_capture()
            elif "gallery" in command or "معرض" in command:
                self.selected_input_method = "gallery"
                if self.selected_language == "Arabic":
                    self.voice_status.setText("فتح المعرض...")
                    self.statusBar().showMessage("فتح المعرض...")
                    self.speak_text("فتح المعرض")
                else:
                    self.voice_status.setText("Opening gallery...")
                    self.statusBar().showMessage("Gallery opening...")
                    self.speak_text("Opening gallery")
                self.open_gallery_for_voice()
        
        elif self.voice_state == "capturing":
            if "capture" in command or "التقط" in command:
                self.capture_image_voice()
        
        # Handle stop command during audio playback
        if ("stop" in command or "توقف" in command) and self.is_playing_audio:
            self.stop_playback()
    
    def start_camera_for_capture(self):
        """Start camera for voice-controlled capture."""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.speak_text("Error: Could not open camera")
                self.reset_to_start()
                return
                
            self.camera_active = True
            self.voice_state = "capturing"
            self.camera_timer.start(30)
            if self.selected_language == "Arabic":
                self.voice_status.setText("الكاميرا نشطة - قل 'التقط' عندما تكون مستعداً")
            else:
                self.voice_status.setText("Camera active - Say 'capture' when ready")
            
        except Exception as e:
            self.speak_text(f"Camera error: {str(e)}")
            self.reset_to_start()
    
    def capture_image_voice(self):
        """Capture image via voice command."""
        if not self.cap or not self.camera_active:
            if self.selected_language == "Arabic":
                self.speak_text("الكاميرا ليست نشطة")
            else:
                self.speak_text("Camera not active")
            self.reset_to_start()
            return
            
        ret, frame = self.cap.read()
        if not ret:
            if self.selected_language == "Arabic":
                self.speak_text("فشل في التقاط الصورة")
            else:
                self.speak_text("Failed to capture image")
            self.reset_to_start()
            return
            
        # Save the captured image to a temporary file
        temp_dir = tempfile.gettempdir()
        temp_file = os.path.join(temp_dir, "visionpal_capture.jpg")
        cv2.imwrite(temp_file, frame)
        
        # Store the current image path
        self.current_image_path = temp_file
        
        # Stop the camera
        self.stop_camera()
        
        # Display the captured image
        self.display_image(temp_file)
        
        # Process the image
        self.voice_state = "processing"
        if self.selected_language == "Arabic":
            self.voice_status.setText("معالجة الصورة...")
            self.speak_text("تم التقاط الصورة. معالجة...")
        else:
            self.voice_status.setText("Processing image...")
            self.speak_text("Image captured. Processing...")
        self.process_image_voice()
    
    def open_gallery_for_voice(self):
        """Open gallery for voice control."""
        file_dialog = QFileDialog()
        image_path, _ = file_dialog.getOpenFileName(
            self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if image_path:
            self.current_image_path = image_path
            self.display_image(image_path)
            self.voice_state = "processing"
            if self.selected_language == "Arabic":
                self.voice_status.setText("معالجة الصورة...")
                self.speak_text("تم اختيار الصورة. معالجة...")
            else:
                self.voice_status.setText("Processing image...")
                self.speak_text("Image selected. Processing...")
            self.process_image_voice()
        else:
            if self.selected_language == "Arabic":
                self.speak_text("لم يتم اختيار صورة")
            else:
                self.speak_text("No image selected")
            self.reset_to_start()
    
    def process_image_voice(self):
        """Process image with voice control."""
        if not self.current_image_path:
            if self.selected_language == "Arabic":
                self.speak_text("لا توجد صورة متاحة")
            else:
                self.speak_text("No image available")
            self.reset_to_start()
            return
            
        prompt = self.available_languages[self.selected_language]["default_prompt"]
        if self.selected_language == "Arabic":
            self.statusBar().showMessage("تحليل الصورة...")
            self.description_text.setText("تحليل الصورة، يرجى الانتظار...")
        else:
            self.statusBar().showMessage("Analyzing image...")
            self.description_text.setText("Analyzing image, please wait...")
        
        # Run in a separate thread
        threading.Thread(
            target=self._process_image_thread,
            args=(self.current_image_path, prompt),
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
        """Update UI with the image description and read it aloud."""
        self.description_text.setText(description)
        self.voice_status.setText("Analysis complete - Say 'start' to begin again")
        self.statusBar().showMessage("Analysis complete - Say 'start' to begin again")
        
        # Read description aloud
        self.speak_text(description)
        
        # Reset to start state after a delay
        QTimer.singleShot(2000, self.reset_to_start)
    
    def update_error(self, error_msg):
        """Update UI with error message."""
        self.description_text.setText(error_msg)
        self.speak_text("Error occurred during analysis")
        self.reset_to_start()
    
    def reset_to_start(self):
        """Reset the application to the start state."""
        self.voice_state = "waiting_for_start"
        self.selected_language = None
        self.selected_input_method = None
        self.current_image_path = None
        self.voice_status.setText('Say "start" to begin')
        self.statusBar().showMessage("Ready - Say 'start' to begin")
        
        # Stop camera if active
        if self.camera_active:
            self.stop_camera()
    
    def stop_camera(self):
        """Stop the camera."""
        self.camera_timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.camera_active = False
    
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
        scaled_pixmap = QPixmap.fromImage(q_image).scaled(
            self.image_preview.width(), 
            self.image_preview.height(),
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        self.image_preview.setPixmap(scaled_pixmap)
    
    def display_image(self, image_path):
        """Display the selected image in the UI."""
        pixmap = QPixmap(image_path)
        scaled_pixmap = pixmap.scaled(
            self.image_preview.width(), 
            self.image_preview.height(),
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        self.image_preview.setPixmap(scaled_pixmap)
    
    def speak_text(self, text):
        """Convert text to speech and play it."""
        try:
            # Create a uniquely named temporary file
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
                temp_path = tmp_file.name
            
            # Get the language code
            lang_code = self.available_languages[self.selected_language]["code"] if self.selected_language else "en"
            
            # Generate speech
            tts = gTTS(text=text, lang=lang_code, slow=False)
            tts.save(temp_path)
            
            # Play the audio
            pygame.mixer.music.load(temp_path)
            pygame.mixer.music.play()
            
            # Set up a timer to delete the file after playback completes
            QTimer.singleShot(10000, lambda: self.cleanup_temp_file(temp_path))
            
        except Exception as e:
            print(f"Speech error: {str(e)}")

    def cleanup_temp_file(self, file_path):
        """Clean up temporary audio file."""
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except:
            pass

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