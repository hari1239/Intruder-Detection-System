import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import cv2
import os
import numpy as np
import face_recognition

class TrainFaceModelApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition Model Training")
        self.root.geometry("400x200")

        self.name_var = tk.StringVar()
        self.image_folder_path = ""

        # Labels and Entry for Name and Image Folder
        tk.Label(root, text="Enter Your Name:").pack(pady=5)
        tk.Entry(root, textvariable=self.name_var).pack(pady=5)
        tk.Button(root, text="Select Image Folder", command=self.browse_image_folder).pack(pady=10)

        # Buttons for Training and Closing
        tk.Button(root, text="Train", command=self.train_model).pack(pady=5)
        tk.Button(root, text="Close", command=root.destroy).pack(pady=5)

    def browse_image_folder(self):
        self.image_folder_path = filedialog.askdirectory()
        if self.image_folder_path:
            print(f"Selected Image Folder: {self.image_folder_path}")

    def train_model(self):
        name = self.name_var.get().strip()

        if not name or not self.image_folder_path:
            tk.messagebox.showwarning("Error", "Please enter your name and select an image folder.")
            return

        known_face_encodings = []
        known_face_names = []

        for filename in os.listdir(self.image_folder_path):
            if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
                image_path = os.path.join(self.image_folder_path, filename)
                user_image = face_recognition.load_image_file(image_path)

                try:
                    user_face_encoding = face_recognition.face_encodings(user_image)[0]
                    known_face_encodings.append(user_face_encoding)
                    known_face_names.append(name)
                    print(f"Face encoded for {name} from {filename}")
                except IndexError:
                    print(f"No face found in {filename}")

        # Save the trained model
        np.savez(f"Final\{name}_model.npz", known_face_encodings=np.array(known_face_encodings), known_face_names=np.array(known_face_names))
        print("Model saved successfully.")

if __name__ == "__main__":
    root = tk.Tk()
    app = TrainFaceModelApp(root)
    root.mainloop()
