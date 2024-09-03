# Conversational-Chatbot
## Overview

This project integrates Google AI services into a Gradio application, featuring:

- **Text Chat:** Generate responses based on text input using a generative AI model.
- **Image Analysis:** Analyze images with prompts using the AI model.
- **Voice Interaction:** Real-time voice interaction with speech recognition and text-to-speech synthesis.

## Prerequisites

Make sure you have the following installed:

- Python 3.7+
- [Pip](https://pip.pypa.io/en/stable/) for managing Python packages

## Setup

1. **Clone the Repository**

   ```bash
   git clone https://github.com/GudisaSandeep/Conversational-Chatbot/tree/main
   cd Conversational-Chatbot
   ```

2. **Create and Activate a Virtual Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Create a `.env` File**

   Copy the `.env.example` file to `.env` and set your API key.

   ```bash
   cp .env.example .env
   ```

   Edit the `.env` file to include your API key:

   ```env
   API_KEY=your_api_key_here
   ```

## Usage

1. **Start the Application**

   Run the following command to start the Gradio app:

   ```bash
   python main2.py
   ```

2. **Access the Application**

   Open your browser and navigate to `http://localhost:7860` to interact with the application.

### Interface Description

- **Text Chat Tab**
  - **Input:** Enter a prompt in the textbox and click "Generate!".
  - **Output:** Displays the AI-generated response in the chatbot.

- **Image Analysis Tab**
  - **Input:** Upload an image and enter a prompt.
  - **Output:** Shows the AI's analysis of the image based on the prompt.

- **Voice Interaction Tab**
  - **Start Voice Interaction:** Click the button to start listening for voice input and receive responses.
  - **Stop Voice Interaction:** Click the button to stop listening.
  - **Status:** Displays the current status of voice interaction.
  - **Conversation:** Shows the ongoing conversation between you and the AI.

## Commands

### Application Commands

- **Start Application:**

  ```bash
  python main2.py
  ```

- **Stop Application:**

  Use `Ctrl+C` in the terminal where the app is running.

### Development Commands

- **Install Dependencies:**

  ```bash
  pip install -r requirements.txt
  ```

- **Create Virtual Environment:**

  ```bash
  python -m venv venv
  ```

- **Activate Virtual Environment:**

  ```bash
  source venv/bin/activate  # On Windows use `venv\Scripts\activate`
  ```



## Contributing

Contributions are welcome! Please submit issues or pull requests to enhance the project.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

