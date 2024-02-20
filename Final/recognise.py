import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import cv2
import numpy as np
import face_recognition

class RecognizeFacesApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition")
        self.root.geometry("300x150")

        # Load the trained model
        self.model_filename = filedialog.askopenfilename(title="Select Model File", filetypes=[("Model files", "*.npz")])
        if not self.model_filename:
            tk.messagebox.showwarning("Error", "Please select a valid model file.")
            self.root.destroy()
            return

        self.load_model()

        # Button for Recognition and Closing
        tk.Button(root, text="Recognize", command=self.recognize_faces).pack(pady=20)
        tk.Button(root, text="Close", command=root.destroy).pack(pady=10)

    def load_model(self):
        model_data = np.load(self.model_filename)
        self.known_face_encodings = model_data['known_face_encodings']
        self.known_face_names = model_data['known_face_names']

    def recognize_faces(self):
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()

            face_locations = face_recognition.face_locations(frame)
            face_encodings = face_recognition.face_encodings(frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)

                name = "Unknown"

                if True in matches:
                    first_match_index = matches.index(True)
                    name = self.known_face_names[first_match_index]

                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, color, 1)

                if name != "Unknown":
                    print(f"Recognized: {name}")

            cv2.imshow('Face Recognition', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    root = tk.Tk()
    app = RecognizeFacesApp(root)
    root.mainloop()
