import gradio as gr
import os
import json
from dotenv import load_dotenv
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from huggingface_hub import login
from datetime import datetime
import numpy as np
import torch
from gtts import gTTS
import tempfile
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import torch

# Load environment variables from .env file
load_dotenv()
login(token="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")

# File paths for storing model configurations and chat history
MODEL_CONFIG_FILE = "model_config.json"
CHAT_HISTORY_FILE = "chat_history.json"

# Load model configurations from a JSON file (if exists)
def load_model_config():
    if os.path.exists(MODEL_CONFIG_FILE):
        with open(MODEL_CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {
        "gpt-4": {

            "endpoint": "https://rogerkoranteng.openai.azure.com/openai/deployments/gpt-4/chat/completions?api-version=2024-08-01-preview",
            "api_key": os.getenv("GPT4_API_KEY"),
            "model_path": None  # No model path for API models
        },
        "gpt-4o": {
            "endpoint": "https://rogerkoranteng.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-08-01-preview",
            "api_key": os.getenv("GPT4O_API_KEY"),
            "model_path": None
        },
        "gpt-35-turbo": {
            "endpoint": "https://rogerkoranteng.openai.azure.com/openai/deployments/gpt-35-turbo/chat/completions?api-version=2024-08-01-preview",
            "api_key": os.getenv("GPT35_TURBO_API_KEY"),
            "model_path": None
        },
        "gpt-4-32k": {
            "endpoint": "https://roger-m38orjxq-australiaeast.openai.azure.com/openai/deployments/gpt-4-32k/chat/completions?api-version=2024-08-01-preview",
            "api_key": os.getenv("GPT4_32K_API_KEY"),
            "model_path": None
        }
    }

predefined_messages = {
    "feeling_sad": "Hello, I am feeling sad today, what should I do?",
    "Nobody likes me": "Hello, Sage. I feel like nobody likes me. What should I do?",
    'Boyfriend broke up': "Hi Sage, my boyfriend broke up with me. I'm feeling so sad. What should I do?",
    'I am lonely': "Hi Sage, I am feeling lonely. What should I do?",
    'I am stressed': "Hi Sage, I am feeling stressed. What should I do?",
    'I am anxious': "Hi Sage, I am feeling anxious. What should I do?",
}

# Save model configuration to JSON
def save_model_config():
    with open(MODEL_CONFIG_FILE, 'w') as f:
        json.dump(model_config, f, indent=4)

# Load chat history from a JSON file
def load_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, 'r') as f:
            return json.load(f)
    return []

# Save chat history to a JSON file
def save_chat_history(chat_history):
    with open(CHAT_HISTORY_FILE, 'w') as f:
        json.dump(chat_history, f, indent=4)

# Define model configurations
model_config = load_model_config()

# Function to dynamically add downloaded model to model_config
def add_downloaded_model(model_name, model_path):
    model_config[model_name] = {
        "endpoint": None,
        "model_path": model_path,
        "api_key": None
    }
    save_model_config()
    return list(model_config.keys())

# Function to download model from Hugging Face synchronously
def download_model(model_name):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model_path = f"./models/{model_name}"
        os.makedirs(model_path, exist_ok=True)
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        updated_models = add_downloaded_model(model_name, model_path)
        return f"Model '{model_name}' downloaded and added.", updated_models
    except Exception as e:
        return f"Error downloading model '{model_name}': {e}", list(model_config.keys())

# Chat function using the selected model
def generate_response(model_choice, user_message, chat_history):
    model_info = model_config.get(model_choice)
    if not model_info:
        return "Invalid model selection. Please choose a valid model.", chat_history

    chat_history.append({"role": "user", "content": user_message})
    headers = {"Content-Type": "application/json"}

    # Check if the model is an API model (it will have an endpoint)
    if model_info["endpoint"]:
        if model_info["api_key"]:
            headers["api-key"] = model_info["api_key"]

        data = {"messages": chat_history, "max_tokens": 1500, "temperature": 0.7}

        try:
            # Send request to the API model endpoint
            response = requests.post(model_info["endpoint"], headers=headers, json=data)
            response.raise_for_status()
            assistant_message = response.json()['choices'][0]['message']['content']
            chat_history.append({"role": "assistant", "content": assistant_message})
            save_chat_history(chat_history)  # Save chat history to JSON
        except requests.exceptions.RequestException as e:
            assistant_message = f"Error: {e}"
            chat_history.append({"role": "assistant", "content": assistant_message})
            save_chat_history(chat_history)
    else:
        # If it's a local model, load the model and tokenizer from the local path
        model_path = model_info["model_path"]
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(model_path)

            inputs = tokenizer(user_message, return_tensors="pt")
            outputs = model.generate(inputs['input_ids'], max_length=500, num_return_sequences=1)
            assistant_message = tokenizer.decode(outputs[0], skip_special_tokens=True)

            chat_history.append({"role": "assistant", "content": assistant_message})
            save_chat_history(chat_history)
        except Exception as e:
            assistant_message = f"Error loading model locally: {e}"
            chat_history.append({"role": "assistant", "content": assistant_message})
            save_chat_history(chat_history)

    # Convert the assistant message to audio
    tts = gTTS(assistant_message)
    audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(audio_file.name)

    return chat_history, audio_file.name

# Function to format chat history with custom bubble styles
def format_chat_bubble(history):
    formatted_history = ""
    for message in history:
        timestamp = datetime.now().strftime("%H:%M:%S")
        if message["role"] == "user":
            formatted_history += f'''
                <div class="user-bubble">
                    <strong>Me:</strong> {message["content"]}
                </div>
            '''
        else:
            formatted_history += f'''
                <div class="assistant-bubble">
                    <strong>Sage:</strong> {message["content"]}
                </div>
            '''
    return formatted_history

tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

def transcribe(audio):
    if audio is None:
        return "No audio input received."

    sr, y = audio

    # Convert to mono if stereo
    if y.ndim > 1:
        y = y.mean(axis=1)

    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

    # Tokenize the audio
    input_values = tokenizer(y, return_tensors="pt", sampling_rate=sr).input_values

    # Perform inference
    with torch.no_grad():
        logits = model(input_values).logits

    # Decode the logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.decode(predicted_ids[0])

    return transcription

# Create the Gradio interface
with gr.Blocks() as interface:
    gr.Markdown("## Chat with Sage - Your Mental Health Advisor")

    with gr.Tab("Model Management"):
        with gr.Tabs():
            with gr.TabItem("Model Selection"):
                gr.Markdown("### Select Model for Chat")
                model_dropdown = gr.Dropdown(choices=list(model_config.keys()), label="Choose a Model", value="gpt-4",
                                             allow_custom_value=True)
                status_textbox = gr.Textbox(label="Model Selection Status", value="Selected model: gpt-4")
                model_dropdown.change(lambda model: f"Selected model: {model}", inputs=model_dropdown,
                                      outputs=status_textbox)

            with gr.TabItem("Download Model"):  # Sub-tab for downloading models
                gr.Markdown("### Download a Model from Hugging Face")
                model_name_input = gr.Textbox(label="Enter Model Name from Hugging Face (e.g. nvidia/Llama-3.1-Nemotron-70B-Instruct-HF)")
                download_button = gr.Button("Download Model")
                download_status = gr.Textbox(label="Download Status")

                # Model download synchronous handler
                def on_model_download(model_name):
                    download_message, updated_models = download_model(model_name)
                    # Trigger the dropdown update to show the newly added model
                    return download_message, gr.update(choices=updated_models, value=updated_models[-1])

                download_button.click(on_model_download, inputs=model_name_input,
                                      outputs=[download_status, model_dropdown])

                refresh_button = gr.Button("Refresh Model List")
                refresh_button.click(lambda: gr.update(choices=list(model_config.keys())), inputs=[],
                                     outputs=model_dropdown)

    with gr.Tab("Chat Interface"):
        gr.Markdown("### Chat with Sage")

        # Chat history state for tracking conversation
        chat_history_state = gr.State(load_chat_history())  # Load existing chat history

        # Add initial introduction message
        if not chat_history_state.value:
            chat_history_state.value.append({"role": "assistant", "content": "Hello, I am Sage. How can I assist you today?"})

        chat_display = gr.HTML(label="Chat", value=format_chat_bubble(chat_history_state.value), elem_id="chat-display")

        user_message = gr.Textbox(placeholder="Type your message here...", label="Your Message")
        send_button = gr.Button("Send Message")

        # Predefined message buttons
        predefined_buttons = [gr.Button(value=msg) for msg in predefined_messages.values()]

        # Real-time message updating
        def update_chat(model_choice, user_message, chat_history_state):
            chat_history, audio_file = generate_response(model_choice, user_message, chat_history_state)
            formatted_chat = format_chat_bubble(chat_history)
            return formatted_chat, chat_history, audio_file

        send_button.click(
            update_chat,
            inputs=[model_dropdown, user_message, chat_history_state],
            outputs=[chat_display, chat_history_state, gr.Audio(autoplay=True)]
        )

        send_button.click(lambda: "", None, user_message)  # Clears the user input after sending

        # Add click events for predefined message buttons
        for button, message in zip(predefined_buttons, predefined_messages.values()):
            button.click(
                update_chat,
                inputs=[model_dropdown, gr.State(message), chat_history_state],
                outputs=[chat_display, chat_history_state, gr.Audio(autoplay=True)]
            )

    with gr.Tab("Speech Interface"):
        gr.Markdown("### Speak with Sage")

        audio_input = gr.Audio(type="numpy")
        transcribe_button = gr.Button("Transcribe")
        transcribed_text = gr.Textbox(label="Transcribed Text")

        transcribe_button.click(
            transcribe,
            inputs=audio_input,
            outputs=transcribed_text
        )

        send_speech_button = gr.Button("Send Speech Message")

        send_speech_button.click(
            update_chat,
            inputs=[model_dropdown, transcribed_text, chat_history_state],
            outputs=[chat_display, chat_history_state, gr.Audio(autoplay=True)]
        )

    # Add custom CSS for scrolling chat box and bubbles
    interface.css = """
        #chat-display {
            max-height: 500px;
            overflow-y: auto;
            padding: 10px;
            background-color: #1a1a1a;
            border-radius: 10px;
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
            scroll-behavior: smooth;
        }

        /* User message style - text only */
        .user-bubble {
            color: #ffffff;  /* Text color for the user */
            padding: 8px 15px;
            margin: 8px 0;
            word-wrap: break-word;
            align-self: flex-end;
            font-size: 14px;
            position: relative;
            max-width: 70%;  /* Make the bubble width dynamic */
            border-radius: 15px;
            background-color: #121212;  /* Light cyan background for the user */
            transition: color 0.3s ease;
        }

        /* Assistant message style - text only */
        .assistant-bubble {
            color: #ffffff;  /* Text color for the assistant */
            padding: 8px 15px;
            margin: 8px 0;
            word-wrap: break-word;
            align-self: flex-start;
            background-color: #2a2a2a;
            font-size: 14px;
            position: relative;
            max-width: 70%;
            transition: color 0.3s ease;
        }

    """
proxy_prefix = os.environ.get("PROXY_PREFIX")
# Launch the Gradio interface
interface.launch(server_name="0.0.0.0", server_port=8080, root_path=proxy_prefix, share=True)
