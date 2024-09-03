# Conversational-Chatbot
This Chatbot give Text Response,Analyze the images and answer queries regarding the image and it can communicate through microphone
Here's a README file for your project. It includes an overview, setup instructions, usage, and a brief description of each component. You might want to adjust some sections based on your specific needs or updates to the code.

---



- **Text Chat:** Interact with a generative AI model for generating responses based on text input.
- **Image Analysis:** Analyze images by providing a prompt to the AI.
- **Voice Interaction:** Real-time voice interaction using speech recognition and text-to-speech synthesis.

## Prerequisites

Ensure you have the following installed:

- Python 3.7+
- [Pip](https://pip.pypa.io/en/stable/) for managing Python packages

## Setup

1. **Clone the Repository**

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install Dependencies**

   Create a virtual environment and install the required packages.

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

3. **Create a `.env` File**

   Copy the `.env.example` file to `.env` and add your API key.

   ```bash
   cp .env.example .env
   ```

   Edit the `.env` file to include your API key:

   ```env
   API_KEY=your_api_key_here
   ```

## Usage

Run the application using:

```bash
python app.py
```

Open your browser and navigate to `http://localhost:7860` to interact with the Gradio interface.

### Text Chat

- **Input:** Enter a text prompt.
- **Output:** Generates a response from the AI model.

### Image Analysis

- **Input:** Upload an image and provide a prompt.
- **Output:** Returns the analysis based on the prompt and image.

### Voice Interaction

- **Start Voice Interaction:** Begins listening for voice input and provides spoken responses.
- **Stop Voice Interaction:** Stops the voice interaction.
- **Status:** Displays the current status of voice interaction.
- **Conversation:** Shows the conversation history between you and the AI.

## Code Description

- **`text_chat(text)`**: Generates a text response from the AI model based on the provided prompt.
- **`image_analysis(image, prompt)`**: Analyzes an image with a given prompt using the AI model.
- **`VoiceInteraction` class**: Handles voice interaction using speech recognition and text-to-speech.
  - **`listen_and_respond()`**: Listens to the microphone, processes the voice input, and generates a response.
  - **`start()`**: Starts the voice interaction in a new thread.
  - **`stop()`**: Stops the voice interaction and joins the thread.
- **Gradio Interface**: Provides tabs for text chat, image analysis, and voice interaction.

## Contributing

Feel free to submit issues or pull requests to improve the project.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Make sure to replace placeholders like `<repository-url>` and `<repository-directory>` with actual values relevant to your project.
