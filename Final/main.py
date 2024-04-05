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
import smtplib
from email.mime.text import MIMEText
import librosa
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import pyaudio
import wave
from datetime import datetime  # Added for timestamp in image names
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage


#image_path=""
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
# def send_email(subject, body):
#     # Email configuration
#     sender_email = 'lifesorted.com@gmail.com'
#     receiver_email = 'harinib200319@gmail.com'  # Change this to your desired recipient email
#     password = 'cgrs apja ghon gfgp'

#     # Create the email content
#     message = MIMEMultipart()
#     #message = MIMEText(body)
#     message['Subject'] = subject
#     message['From'] = sender_email
#     message['To'] = receiver_email
#     message.attach(MIMEText(intruder_body, 'plain'))
#     #image_path = f"Final/intruders_images/intruder_{timestamp}.jpg"
#     with open(image_path, 'rb') as img_file:
#         image_data = img_file.read()
#         image = MIMEImage(image_data)
#         image.add_header('Content-Disposition', 'attachment', filename=os.path.basename(image_path))
#         message.attach(image)

#     try:
#         # Connect to the SMTP server
#         server = smtplib.SMTP('smtp.gmail.com', 587)
#         server.starttls()

#         # Log in to your email account
#         server.login(sender_email, password)

#         # Send the email
#         server.sendmail(sender_email, receiver_email, message.as_string())

#         # Close the connection
#         server.quit()
#         print("Email sent successfully!")
#     except Exception as e:
#         print(f"Error sending email: {e}")

# Example usage
intruder_subject = "Intruder Detected"
intruder_body = "An intruder has been detected. Please take appropriate action."

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
    score_face_haarcascades = detect_face_haarcascades(frame)
    score_face_recognition = detect_face_face_recognition(frame)
    score_mask = detect_mask(frame)

    return score_face_haarcascades, score_face_recognition, score_mask

# Function to play an alarm sound
def play_alarm():
    alarm_sound = AudioSegment.from_file("Final\Intruder_detected_1.wav", format="wav")
    play(alarm_sound)
def play_alarm1():
    alarm_sound = AudioSegment.from_file(r"Final\\User_Recognized_Unlockin_1.wav", format="wav")
    play(alarm_sound)
# Function to save intruder images
def save_intruder_image(frame):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    image_path = f"Final/intruders_images/intruder_{timestamp}.jpg"
    cv2.imwrite(image_path, frame)
    print(f"Intruder image saved: {image_path}")

    # Email configuration
    sender_email = 'lifesorted.com@gmail.com'
    receiver_email = 'harinib200319@gmail.com'  # Change this to your desired recipient email
    password = 'cgrs apja ghon gfgp'

    # Create the email content
    message = MIMEMultipart()
    #message = MIMEText(body)
    message['Subject'] = intruder_subject
    message['From'] = sender_email
    message['To'] = receiver_email
    message.attach(MIMEText(intruder_body, 'plain'))
    #image_path = f"Final/intruders_images/intruder_{timestamp}.jpg"
    with open(image_path, 'rb') as img_file:
        image_data = img_file.read()
        image = MIMEImage(image_data)
        image.add_header('Content-Disposition', 'attachment', filename=os.path.basename(image_path))
        message.attach(image)

    try:
        # Connect to the SMTP server
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()

        # Log in to your email account
        server.login(sender_email, password)

        # Send the email
        server.sendmail(sender_email, receiver_email, message.as_string())

        # Close the connection
        server.quit()
        print("Email sent successfully!")
    except Exception as e:
        print(f"Error sending email: {e}")

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

        # Draw rectangle and label on the frame
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        return name  # Return the recognized name or "Unknown" if not recognized

    return "Unknown"  # Return "Unknown" if no face is found

# Open the webcam
cap = cv2.VideoCapture(0)

# Load face recognition model
known_face_encodings, known_face_names = load_face_model("Final\Gouri_model.npz")

# Include the LockingSystem class from program 5 here

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

        if total_score <= score_limit:
            # Start a new thread to play the alarm sound
            threading.Thread(target=play_alarm).start()
            #send_email(intruder_subject, intruder_body)
            # Save intruder image
            save_intruder_image(frame)

            # If the recognized face is known, unlock the system
            if recognized_face != "Unknown":
                threading.Thread(target=play_alarm1).start()
                print("Recognized face: known person")
                print("System unlocked!")
                break  # You can add your unlocking logic here
            break

except KeyboardInterrupt:
    print("Intrusion detection stopped.")
    cap.release()
    cv2.destroyAllWindows()
    lock_system()
