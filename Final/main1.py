import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk  # Added for displaying webcam feed
import face_recognition
from pydub import AudioSegment
from pydub.playback import play
import threading
import os
import librosa
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import pyaudio
import wave
from datetime import datetime  # Added for timestamp in image names

# Function to extract audio features using Librosa
def lock_system():
    root = tk.Tk()
    app = LockingSystem(root)
    root.mainloop()

def extract_audio_features(file_path, duration=3):
    y, sr = librosa.load(file_path, duration=duration)
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

    frames = []
    for i in range(0, int(sample_rate / 1024 * duration)):
        data = stream.read(1024)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    audio.terminate()

    wf = wave.open("temp.wav", 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()

    sound = AudioSegment.from_wav("temp.wav")
    sound.export(file_name, format="mp3")

    os.remove("temp.wav")

    return file_name

# Function to load data and extract features for audio-based intrusion detection
def load_audio_data(target_folder, non_target_folder):
    target_files = librosa.util.find_files(target_folder)
    non_target_files = librosa.util.find_files(non_target_folder)

    target_data = [extract_audio_features(file) for file in target_files]
    non_target_data = [extract_audio_features(file) for file in non_target_files]

    target_labels = np.ones(len(target_data))
    non_target_labels = np.zeros(len(non_target_data))

    X = np.vstack([target_data, non_target_data])
    y = np.concatenate([target_labels, non_target_labels])

    return X, y

# Function to detect face using Haarcascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_face_haarcascades(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    return len(faces)

# Function to detect face using face_recognition
def detect_face_face_recognition(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_recognition.face_locations(rgb_frame)
    return len(faces)

# Function to detect mask
def detect_mask(frame):
    # Implement your mask detection logic here
    # Return 1 for mask detected, 0 for no mask
    return 1

# Function to assign scores based on individual detection methods
def assign_scores(frame):
    score_mask = detect_mask(frame)
    score_audio = 0  # Add logic for audio detection
    score_face_duration = 0

    # Initialize or update the timer based on face detection
    if detect_face_haarcascades(frame) > 0:
        if 'start_time' not in assign_scores.__dict__:
            assign_scores.start_time = datetime.now()
        else:
            elapsed_time = datetime.now() - assign_scores.start_time
            score_face_duration = elapsed_time.total_seconds()

    else:
        assign_scores.start_time = None  # Reset the timer if no face is detected

    # Return the individual scores as a tuple
    return score_mask, score_audio, score_face_duration



# Function to play an alarm sound
def play_alarm(alarm_file):
    alarm_sound = AudioSegment.from_file(alarm_file, format="wav")
    play(alarm_sound)

# Function to save intruder images
def save_intruder_image(frame):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    image_path = f"intruders_images/intruder_{timestamp}.jpg"
    cv2.imwrite(image_path, frame)
    print(f"Intruder image saved: {image_path}")

# Function to load face recognition model
def load_face_model(face_model_path):
    face_model = np.load(face_model_path)
    known_face_encodings = face_model['known_face_encodings']
    known_face_names = face_model['known_face_names']
    return known_face_encodings, known_face_names

# Function to recognize face
def recognize_face(frame, known_face_encodings, known_face_names):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # If a match is found, use the name of the known face
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

            # Play another sound and close the program
            threading.Thread(target=play_alarm, args=("User Recognized Unlockin 1.wav",)).start()
            messagebox.showinfo("Recognition", f"Recognized face: {name}")
            exit()

        # Draw rectangle and label on the frame
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    return "Unknown"  # Return "Unknown" if no face is found

# Open the webcam
cap = cv2.VideoCapture(0)

# Load face recognition model
known_face_encodings, known_face_names = load_face_model("Final\Gouri_model.npz")

# Set intrusion detection score limit
score_limit = 1  # Adjust this value based on your requirements

# Create Tkinter window
root = tk.Tk()
root.title("Webcam Feed")

# Create label to display webcam feed
label = tk.Label(root)
label.pack()

try:
    while True:
        ret, frame = cap.read()

        # Assign scores for individual detection methods
        score_face_haarcascades, score_face_recognition, score_mask = assign_scores(frame)

        # Recognize face
        recognized_face = recognize_face(frame, known_face_encodings, known_face_names)

        # Score aggregation logic (you can customize this based on your requirements)
        total_score = score_face_haarcascades + score_face_recognition + score_mask

        print(f"Total Score: {total_score}")

        # Convert frame to RGB for displaying in Tkinter
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        frame_tk = ImageTk.PhotoImage(frame_pil)

        # Update label with new frame
        label.configure(image=frame_tk)
        label.image = frame_tk

        # Update Tkinter window
        root.update()

except KeyboardInterrupt:
    print("Intrusion detection stopped.")
    cap.release()
    cv2.destroyAllWindows()
    lock_system()

