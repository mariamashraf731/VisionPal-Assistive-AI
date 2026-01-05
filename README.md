# 👁️ VisionPal: AI Assistant for the Visually Impaired

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Model](https://img.shields.io/badge/Model-Llama%203.2%20Vision-orange)
![Framework](https://img.shields.io/badge/App-Streamlit-red)
![Accessibility](https://img.shields.io/badge/Focus-Assistive%20Tech-green)

## 📌 Project Overview
**VisionPal** is an AI-powered assistive application designed to empower visually impaired individuals by providing auditory descriptions of their surroundings. 

The application uses a powerful vision-language model, **`meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo`** via **Together AI**, to analyze images from the user's camera. It provides rich, contextual descriptions of the scene, identifies objects, and can answer follow-up questions to help with navigation and environmental awareness.

This core AI capability, paired with **Text-to-Speech (Google TTS)**, offers a comprehensive and interactive way for users to understand their environment.

## ⚙️ Key Features
* **AI-Powered Scene Understanding:** Leverages the Llama vision model to generate rich descriptions of scenes and answer user questions about the image.
* **Auditory Feedback:** Converts AI-generated descriptions and answers into speech using **gTTS**.
* **Multiple Modes:**
    * **Voice-Activated Mode:** Totally hands-free operation.
    * **Button-Based Mode:** For tactile control when preferred.
* **Web Interface:** Accessible via browser using **Streamlit** for easy deployment and testing.
* **Multi-language Support:** Offers language selection (Arabic/English).
* **Flexible Input:** Supports both live camera feed and gallery image uploads.
* **Voice Interaction:** Includes speech-to-text and text-to-speech capabilities.
* **Noise Reduction:** Features noise-reduced audio input for clearer commands.

## 🛠️ Tech Stack
* **Core Logic:** Python.
* **Computer Vision & AI:** `meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo` via Together AI, OpenCV.
* **Audio Processing:** Google Text-to-Speech (gTTS), PyAudio.
* **Frontend:** Streamlit.

## 🚀 Getting Started

Follow these steps to get VisionPal running on your local machine.

### 1. Installation and Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/mariamashraf731/VisionPal-Assistive-AI.git
    cd VisionPal-Assistive-AI
    ```

2.  **Install dependencies:**
    *(This project's dependencies are listed in `requirements.txt`.)*
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure API Key:**
    Create a `.env` file in the project's root directory and add your Together AI API key. This is required for the AI-powered description feature.

    ```.env
    TOGETHER_API_KEY="your_together_api_key_here"
    ```

### 2. Run the Application

You can run the application in two different modes:

*   **Button-Based Mode (Desktop):**
    ```bash
    python app_button.py
    ```

*   **Streamlit Web App:**
    ```bash
    streamlit run src/app_streamlit.py
    ```

## Configuration
- Adjust language settings
- Customize vision model
- Fine-tune noise reduction parameters

## Troubleshooting
- Ensure all dependencies are installed
- Check microphone permissions
- Verify API keys

## 📄 Documentation
For detailed system architecture and user flow, refer to the [Project Report](docs/VisionPal_report.pdf).


## License
[MIT](https://choosealicense.com/licenses/mit/)
