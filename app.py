from flask import Flask, render_template, request, send_file, jsonify, redirect, url_for
import os
import uuid
import sqlite3
from faster_whisper import WhisperModel
from TTS.api import TTS
import requests
from pydub import AudioSegment

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESPONSE_FOLDER'] = 'responses'
app.config['VOICE_MODEL_FOLDER'] = 'voice_models'
app.config['DB_PATH'] = 'database/voice_chatbot.db'

OPENROUTER_API_KEY = "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"

# Initialize STT
whisper_model = WhisperModel("tiny.en", compute_type="int8")

# ✅ Switch to XTTSv2 for real speaker cloning
tts_model = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False, gpu=False)

# Ensure directories exist
for folder in [app.config['UPLOAD_FOLDER'], app.config['RESPONSE_FOLDER'], app.config['VOICE_MODEL_FOLDER'], 'database']:
    os.makedirs(folder, exist_ok=True)

def init_db():
    with sqlite3.connect(app.config['DB_PATH']) as conn:
        cursor = conn.cursor()
        cursor.execute(''' 
            CREATE TABLE IF NOT EXISTS voice_models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE,
                path TEXT
            )
        ''')
        conn.commit()

init_db()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train-model')
def train_model_page():
    return render_template('train_model.html')

@app.route('/chat')
def chat_page():
    with sqlite3.connect(app.config['DB_PATH']) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM voice_models")
        voices = [row[0] for row in cursor.fetchall()]
    return render_template('chat.html', voices=voices)

@app.route('/train-model', methods=['POST'])
def train_voice_model():
    model_name = request.form.get('model_name')
    files = request.files.getlist('voice_samples')

    model_dir = os.path.join(app.config['VOICE_MODEL_FOLDER'], model_name)
    os.makedirs(model_dir, exist_ok=True)

    print(f"Training new model: {model_name}")
    print(f"Saving to directory: {model_dir}")

    for i, file in enumerate(files):
        original_filename = file.filename
        print(f"Processing file {original_filename}")
        try:
            audio = AudioSegment.from_file(file)
            converted_filename = f"{i}_{os.path.splitext(original_filename)[0]}.wav"
            save_path = os.path.join(model_dir, converted_filename)
            audio.export(save_path, format="wav")
            print(f"Saved converted WAV to: {save_path}")
        except Exception as e:
            print(f"Error converting {original_filename}: {e}")

    with sqlite3.connect(app.config['DB_PATH']) as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT OR IGNORE INTO voice_models (name, path) VALUES (?, ?)", (model_name, model_dir))
        conn.commit()
        print(f"Model {model_name} registered in DB.")
    
    return redirect(url_for('chat_page'))

@app.route('/upload-audio', methods=['POST'])
def upload_audio():
    file = request.files['audio']
    emotion = request.form.get('emotion', 'neutral')
    voice_id = request.form.get('voice_id', 'default')
    prompt_template = request.form.get('prompt-template', 'assistant')  # Capture prompt template

    print(f"Received audio for voice_id: {voice_id} with emotion: {emotion} and template: {prompt_template}")

    filename = str(uuid.uuid4()) + ".wav"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    print(f"Saved uploaded file to: {filepath}")

    if not filepath.endswith(".wav"):
        print(f"Converting {filepath} to .wav")
        sound = AudioSegment.from_file(filepath)
        filepath = filepath.replace(".webm", ".wav")
        sound.export(filepath, format="wav")

    print("Transcribing...")
    segments, _ = whisper_model.transcribe(filepath)
    text = "".join([seg.text for seg in segments])
    print(f"Transcribed text: {text}")

    ai_response_text = get_ai_response(text, prompt_template)  # Pass template to AI response
    response_text = f"[{emotion}] {ai_response_text}"
    print(f"AI Response: {response_text}")

    with sqlite3.connect(app.config['DB_PATH']) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT path FROM voice_models WHERE name = ?", (voice_id,))
        result = cursor.fetchone()

    if not result:
        print(f"Voice model '{voice_id}' not found.")
        return jsonify({"error": "Voice model not found"}), 400

    voice_model_path = result[0]
    print(f"Using voice model from path: {voice_model_path}")

    wav_files = [f for f in os.listdir(voice_model_path) if f.endswith('.wav')]
    print(f"Found {len(wav_files)} training samples.")

    if not wav_files:
        return jsonify({"error": "No .wav file found in the voice model folder"}), 400

    voice_model_wav = os.path.join(voice_model_path, wav_files[0])
    output_path = os.path.join(app.config['RESPONSE_FOLDER'], f"{uuid.uuid4()}.wav")

    try:
        print(f"Generating TTS with reference: {voice_model_wav}")
        tts_model.tts_to_file(
            text=response_text,
            speaker_wav=voice_model_wav,
            language="en",
            file_path=output_path
        )
        print(f"TTS saved at: {output_path}")
        return jsonify({"response_audio": output_path.replace('\\', '/')})
    except Exception as e:
        print(f"TTS Error: {e}")
        return jsonify({"error": "TTS generation failed", "details": str(e)}), 500

@app.route('/get-audio/<filename>')
def get_audio(filename):
    filepath = os.path.join(app.config['RESPONSE_FOLDER'], filename)
    return send_file(filepath, mimetype="audio/wav")

def get_ai_response(user_input, prompt_template):
    # Map templates to specific prompt styles
    template_prompts = {
        "assistant": "You are a helpful assistant. Respond politely and provide useful information.",
        "friend": "You are a friendly companion. Respond in a casual, relaxed manner.",
        "tutor": "You are a knowledgeable tutor. Respond with educational and helpful explanations."
    }

    # Default to "assistant" if no valid template is selected
    prompt = template_prompts.get(prompt_template, template_prompts["assistant"])

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "meta-llama/llama-3-8b-instruct",
        "messages": [{"role": "user", "content": f"{prompt} {user_input}"}],
    }

    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        ai_message = response.json()["choices"][0]["message"]["content"].strip()
        print(f"OpenRouter AI response: {ai_message}")
        return ai_message
    except Exception as e:
        print(f"OpenRouter Error: {e}")
        return "I'm sorry, I couldn't process your request right now."

if __name__ == '__main__':
    app.run(debug=True)
