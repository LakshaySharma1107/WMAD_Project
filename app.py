from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from pydub import AudioSegment
import numpy as np
import librosa
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import os

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests for React Native

# Load your pre-trained model
model = load_model('My_Best_Model.h5')
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Define the segment duration (500 milliseconds)
segment_duration = 500

# Directory to save processed audio files locally
PROCESSED_AUDIO_DIR = 'processed_audio_files'

# Ensure that the directory exists
if not os.path.exists(PROCESSED_AUDIO_DIR):
    os.makedirs(PROCESSED_AUDIO_DIR)

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

            # Save the processed (muted) audio locally
            processed_audio_path = os.path.join(PROCESSED_AUDIO_DIR, f"muted_audio.wav")
            audio.export(processed_audio_path, format="wav")

            return processed_audio_path
        else:
            print("No valid segments to predict.")
            return None
    except Exception as e:
        print(f"Error processing audio file: {e}")
        return None

# File Upload Route
@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    try:
        # Get the audio file from the request
        audio_file = request.files['file']
        audio_file_name = audio_file.filename
        audio = AudioSegment.from_file(audio_file)

        # Process the audio file to mute bad words
        processed_audio_path = process_audio(audio)

        if processed_audio_path:
            return jsonify({"message": "Audio processed successfully", "path": processed_audio_path})
        else:
            return jsonify({"message": "Audio processing failed"}), 400
    except Exception as e:
        print(f"Error during file upload: {e}")
        return jsonify({"message": "Error during file upload"}), 500

# File Download Route
@app.route('/download_audio/<audio_file_name>', methods=['GET'])
def download_audio(audio_file_name):
    try:
        processed_audio_path = os.path.join(PROCESSED_AUDIO_DIR, audio_file_name)
        if os.path.exists(processed_audio_path):
            return send_file(processed_audio_path, as_attachment=True, download_name=audio_file_name)
        else:
            return jsonify({"message": "File not found"}), 404
    except Exception as e:
        print(f"Error during file download: {e}")
        return jsonify({"message": "Error during file download"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Default to 5000 if PORT is not set
    app.run(debug=True, host="0.0.0.0", port=port)
