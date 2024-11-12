from flask import Flask, request, jsonify, send_file, abort
from flask_cors import CORS
from pydub import AudioSegment
import numpy as np
import librosa
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import os
import time
import shutil
import pickle
from vosk import Model, KaldiRecognizer
import wave
import json
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import subprocess

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests for React Native

# Load your pre-trained model
model = load_model('My_Best_Model.h5')

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

model_2 = load_model('fast_text_model.h5')
model_2.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Load the tokenizers and model (assuming they're saved as shown earlier)
with open('ft_word_tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)
with open('ft_char_tokenizer.pkl', 'rb') as handle:
    char_tokenizer = pickle.load(handle)

# Define the segment duration (500 milliseconds)
segment_duration = 500


# Function to extract features from audio segments
def extract_features_from_segment(segment):
    try:
        samples = np.array(segment.get_array_of_samples()).astype(np.float32)
        mfccs = librosa.feature.mfcc(y=samples, sr=segment.frame_rate, n_mfcc=15)
        mfccs = np.mean(mfccs.T, axis=0)
        return mfccs
    except Exception as e:
        print(f"Error processing segment: {e}")
        return None

# Function to process the audio file and mute bad words
def process_audio(audio):
    global audio_file_name
    try:
        bad_word_indices = []  # Store indices of bad word segments
        segments = []  # To store MFCC features for each segment

        # Split the audio into segments and process each segment
        for i in range(0, len(audio), segment_duration):
            segment = audio[i:i + segment_duration]
            mfccs = extract_features_from_segment(segment)
            if mfccs is not None:
                segments.append(mfccs)

        # Prepare the input for the model
        X_test = np.array(segments)

        if len(X_test) > 0:
            # Get predictions from the model
            predictions = model.predict(X_test)
            
            # Check for bad word predictions (Assuming 1 means bad word)
            for i, pred in enumerate(predictions):
                if np.argmax(pred) == 1:  # Assuming bad word is labeled as 1
                    start_time = i * segment_duration
                    end_time = start_time + segment_duration
                    bad_word_indices.append((start_time, end_time))

            # Mute the bad words in the audio
            for start, end in bad_word_indices:
                audio = audio[:start] + AudioSegment.silent(duration=end - start) + audio[end:]

            # Save the processed (muted) audio to a file
            processed_audio_path = f'static/muted_audio_{audio_file_name}'
            audio.export(processed_audio_path, format="wav")
            return processed_audio_path
        else:
            print("No valid segments to predict.")
            return None
    except Exception as e:
        print(f"Error processing audio file: {e}")
        return None
#------------------------------------------Text------------------------------------------
def preprocess_sentence(sentence):
    char_max_length = 15
    max_length = 475
    word_sequence = tokenizer.texts_to_sequences([sentence])
    padded_word_sequence = pad_sequences(word_sequence, maxlen=max_length, padding='post', truncating='post')
    char_sequence = [[char_tokenizer.word_index.get(char, 0) for char in word] for word in sentence.split()]
    char_sequence = pad_sequences(char_sequence, maxlen=char_max_length, padding="post")
    padded_char_sequence = pad_sequences([char_sequence], maxlen=max_length, padding='post', dtype='int32')
    return padded_word_sequence, padded_char_sequence

def predict_bad_words(sentence):
    max_length = 475
    padded_word_sequence, padded_char_sequence = preprocess_sentence(sentence)
    predictions = model_2.predict([padded_word_sequence, padded_char_sequence])  # Feed both sequences
    threshold = 0.5
    predicted_labels = (predictions > threshold).astype(int)[0]
    words = sentence.split()
    bad_words = [word for i, word in enumerate(words[:max_length]) if predicted_labels[i] == 1]
    return bad_words

def convert_mp3_to_wav(mp3_file, wav_file):
    try:
        subprocess.run(['ffmpeg', '-i', mp3_file, '-ac', '1', '-ar', '16000', wav_file], check=True)
    except subprocess.CalledProcessError as e:
        print("Error during MP3 to WAV conversion:", e)

def transcribe_audio_with_timestamps(audio_file, model_path):
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return []
    
    model = Model(model_path)
    if not audio_file.lower().endswith('.wav'):
        wav_file = audio_file.rsplit('.', 1)[0] + '.wav'
        if audio_file.lower().endswith('.mp3'):
            convert_mp3_to_wav(audio_file, wav_file)
            audio_file = wav_file
        else:
            print("Audio file must be in WAV format or MP3 format.")
            return []
    
    try:
        wf = wave.open(audio_file, "rb")
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
            print("Audio file must be WAV format mono PCM.")
            return []
    except Exception as e:
        print(f"Error opening audio file: {e}")
        return []
    
    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)
    results = []
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            results.append(result)
        else:
            rec.PartialResult()
    final_result = json.loads(rec.FinalResult())
    results.append(final_result)
    
    word_timestamps = []
    for result in results:
        if 'result' in result:
            for word_info in result['result']:
                word_timestamps.append({
                    'word': word_info.get('word', ''),
                    'start': word_info.get('start', 0),
                    'end': word_info.get('end', 0)
                })
    return word_timestamps

def mute_bad_words_in_audio(audio_file, bad_words, word_timestamps):
    if not os.path.exists(audio_file):
        print(f"Audio file {audio_file} not found.")
        return None
    
    audio = AudioSegment.from_wav(audio_file)
    for item in word_timestamps:
        if item['word'] in bad_words:
            start_ms = item['start'] * 1000
            end_ms = item['end'] * 1000
            audio = audio[:start_ms] + AudioSegment.silent(duration=(end_ms - start_ms)) + audio[end_ms:]
            

    return audio


    
def process_audio_text(audio_file, model_path):
    try:
        # Step 1: Transcribe audio with timestamps
        word_timestamps = transcribe_audio_with_timestamps(audio_file, model_path)
        if not word_timestamps:
            raise ValueError("No transcribed words found in the audio.")

    except Exception as e:
        print(f"Error in transcription step: {e}")
        return None  # Or handle the error as needed

    try:
        # Step 2: Detect bad words from transcribed text
        transcribed_text = " ".join([item['word'] for item in word_timestamps])
        bad_words = predict_bad_words(transcribed_text)
        if not bad_words:
            print("No bad words detected in the transcription.")
            return audio_file  # Return original file if no bad words are detected

    except Exception as e:
        print(f"Error in bad word detection: {e}")
        return None  # Or handle the error as needed

    try:
        # Step 3: Mute bad words in audio
        muted_file = mute_bad_words_in_audio(audio_file, bad_words, word_timestamps)
        if not muted_file:
            raise ValueError("Failed to create muted audio file.")

    except Exception as e:
        print(f"Error in muting bad words: {e}")
        return None  # Or handle the error as needed

    try:
        # Save the processed (muted) audio to a file
        processed_audio_path = f'static/muted_audio_{audio_file_name}'
        muted_file.export(processed_audio_path, format="wav")
        return processed_audio_path

    except Exception as e:
        print(f"Error in saving the muted audio file: {e}")
        return None  # Or handle the error as needed


    

audio_file_name = "" ########-----------

file_path_glob = ""
# API route to process the audio
@app.route('/process-audio', methods=['POST'])
def process_audio_route():
    global file_path_glob, audio_file_name
    try:
        # print(audio_file)
        # Receive the audio file from the request
        audio_file = request.files.get('audio')
        model_type = request.form.get('model') #gets which model is picked

        print(audio_file)    
        print(audio_file.filename)
        audio_file_name  = audio_file.filename

        if not audio_file:
            return jsonify({'status': 'error', 'message': 'No audio file received'}), 400

        # Determine the file extension and load it
        file_extension = audio_file.filename.split('.')[-1].lower()
        if file_extension not in ['wav', 'mp3']:
            return jsonify({'status': 'error', 'message': 'Invalid file format. Only WAV and MP3 are supported.'}), 400

        # Save the uploaded file temporarily
        temp_file_path = os.path.join('static', audio_file.filename)
        print(temp_file_path)
        audio_file.save(temp_file_path)

        # Load the audio file with pydub
        if file_extension == 'wav':
            audio = AudioSegment.from_wav(temp_file_path)
        else:
            audio = AudioSegment.from_mp3(temp_file_path)

#---------------------------------------> Process the audio file <------------------------------------------------------------------------
        if model_type == 'Audio Model':
            processed_audio_path = process_audio(audio)
        
        else:
        
            processed_audio_path = process_audio_text("static\\"+ audio_file_name,'vosk-model-small-hi-0.22')
            
        file_path_glob = processed_audio_path
        print("Global File Path === ",file_path_glob,audio_file_name,file_path_glob)
        
        if processed_audio_path:
            return jsonify({'processed_audio': f'/{processed_audio_path}'}), 200
        else:
            return jsonify({'status': 'error', 'message': 'Processing failed'}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Serve processed audio files and delete them after serving
@app.route('/static/<filename>')
def serve_and_delete_file(filename):
    global file_path_glob
    # file_path = os.path.join('static', filename)
    # file_path = f"static/muted_{audio_file_name}"
    file_path = file_path_glob
    print(file_path)
    try:
        # Send the file
        response = send_file(file_path)
        print(response)
        return response
    # except FileNotFoundError:
    #     abort(404, description="File not found")
    # except Exception as e:
    except:
        pass
    #     abort(500, description=str(e))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 