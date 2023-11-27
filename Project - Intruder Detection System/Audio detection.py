import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import pyaudio
import wave
from pydub import AudioSegment
from pydub.playback import play
import threading


# Function to extract audio features using Librosa
def extract_features(file_path,duration=3):
    y, sr = librosa.load(file_path,duration=duration)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)



# Function to record audio in real-time and save it in MP3 format
def record_audio(duration=3, sample_rate=44100, file_name="recorded_audio.mp3"):
    audio = pyaudio.PyAudio()

    stream = audio.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=1024)

    #print("Recording...")
    frames = []
    for i in range(0, int(sample_rate / 1024 * duration)):
        data = stream.read(1024)
        frames.append(data)

    #print("Finished recording.")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    wf = wave.open("temp.wav", 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()

    # Load the temporary WAV file and export it to MP3
    sound = AudioSegment.from_wav("temp.wav")
    sound.export(file_name, format="mp3")
    
    #Remove the temporary WAV file
    import os
    os.remove("temp.wav")

    return file_name


# Function to load data and extract features
def load_data(target_folder, non_target_folder):
    target_files = librosa.util.find_files(target_folder)
    non_target_files = librosa.util.find_files(non_target_folder)

    target_data = [extract_features(file) for file in target_files]
    non_target_data = [extract_features(file) for file in non_target_files]

    # Labeling data
    target_labels = np.ones(len(target_data))
    non_target_labels = np.zeros(len(non_target_data))

    # Concatenating target and non-target data
    X = np.vstack([target_data, non_target_data])
    y = np.concatenate([target_labels, non_target_labels])

    return X, y

# Define your paths to the target and non-target audio files
target_folder = r"Project - Intruder Detection System\Target_Set_Glass shattering Audio"
non_target_folder = r"Project - Intruder Detection System\Non_Target_Set_Glass shattering Audio"

# Load data and extract features
X, y = load_data(target_folder, non_target_folder)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm_classifier = SVC(kernel='linear',C=1)
svm_classifier.fit(X_train_scaled, y_train)

# Make predictions
y_pred = svm_classifier.predict(X_test_scaled)

# Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy:.2f}")
# print("Classification Report:\n", classification_report(y_test, y_pred))

stop_alarm = False

# Function to play an alarm sound
def play_alarm():
    alarm_sound = AudioSegment.from_file(r"Project - Intruder Detection System\Alarm.mp3", format="mp3")
    play(alarm_sound)

try:
    while True:
        # Record real-time audio
        recorded_audio_file = record_audio()

        # Extract features from the real-time audio
        input_features = extract_features(recorded_audio_file)

        # Normalize the features using the scaler
        input_features_scaled = scaler.transform([input_features])

        # Make real-time predictions
        prediction = svm_classifier.predict(input_features_scaled)

        print(f"The model predicts: {prediction}")

        if prediction == 1 and not stop_alarm:
            # Start a new thread to play the alarm sound
            threading.Thread(target=play_alarm).start()

        # elif prediction == 0:
        #     # Stop the alarm by setting the stop_alarm flag
        #     stop_alarm = True

except KeyboardInterrupt:
    print("Real-time processing stopped.")
    # Stop the alarm if it's still playing
    stop_alarm = True