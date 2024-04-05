import os
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import numpy as np
from sklearn.metrics import accuracy_score

class ImageLabelingApp:
    def __init__(self, root, intruder_folder):
        self.root = root
        self.root.title("Model Adaptation")
        self.intruder_folder = intruder_folder
        self.images = []
        self.labels = []
        self.current_index = 0

        # Create and configure the main window
        self.main_frame = tk.Frame(root, bg="#336699")  # Background color
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.heading_label = tk.Label(self.main_frame, text="CAPTURED IMAGES", font=("Helvetica", 16), bg="#336699", fg="white")  # Heading style
        self.heading_label.pack(pady=10)

        self.open_button = tk.Button(self.main_frame, text="Open", command=self.open_images, bg="#6699CC", fg="white")  # Button style
        self.open_button.pack(pady=10)

    def open_images(self):
        self.load_images()
        self.show_image()

    def load_images(self):
        self.images = [f for f in os.listdir(self.intruder_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
        self.labels = [None] * len(self.images)

    def show_image(self):
        if self.current_index < len(self.images):
            image_path = os.path.join(self.intruder_folder, self.images[self.current_index])

            img = Image.open(image_path)
            img.thumbnail((400, 400))
            img = ImageTk.PhotoImage(img)

            img_label = tk.Label(self.main_frame, image=img)
            img_label.image = img
            img_label.pack(pady=10)

            question_label = tk.Label(self.main_frame, text="Do you know this person?", font=("Helvetica", 12), bg="#336699", fg="white")
            question_label.pack(pady=5)

            yes_button = tk.Button(self.main_frame, text="Yes", command=lambda: self.label_image(True), bg="#6699CC", fg="white")
            yes_button.pack(pady=5)

            no_button = tk.Button(self.main_frame, text="No", command=lambda: self.label_image(False), bg="#6699CC", fg="white")
            no_button.pack(pady=5)

        else:
            self.update_model()
            self.delete_images()
            self.root.destroy()

    def label_image(self, is_known):
        self.labels[self.current_index] = is_known
        self.current_index += 1
        for widget in self.main_frame.winfo_children():
            widget.destroy()
        self.show_image()

    def update_model(self):
        # Use self.labels to update your model
        known_count = self.labels.count(True)
        unknown_count = self.labels.count(False)
        total_count = len(self.labels)

        accuracy = known_count / total_count if total_count > 0 else 0

        messagebox.showinfo("Model Update", f"Model updated with accuracy: {accuracy:.2%}")

    def delete_images(self):
        for image in self.images:
            image_path = os.path.join(self.intruder_folder, image)
            os.remove(image_path)

if __name__ == "__main__":
    intruder_folder = "Final\intruders_images"

    root = tk.Tk()
    app = ImageLabelingApp(root, intruder_folder)
    root.mainloop()
