import tkinter as tk
import subprocess
import os

class SafeFaceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SafeFace - Intruder Detection System")

        # Create a larger canvas
        canvas = tk.Canvas(root, height=300, width=400, bg="lightblue")
        canvas.pack()

        # Create buttons with colors and animation
        button1 = tk.Button(root, text="Surveillance", command=self.run_program1, bg="green", activebackground="darkgreen")
        button2 = tk.Button(root, text="Add faces", command=self.run_program2, bg="blue", activebackground="darkblue")
        button3 = tk.Button(root, text="Model adaptation", command=self.run_program3, bg="red", activebackground="darkred")
        # button4 = tk.Button(root, text="Run Program 4", command=self.run_program4, bg="yellow", activebackground="darkyellow")

        # Pack buttons
        canvas.create_window(200, 80, window=button1)
        canvas.create_window(200, 150, window=button2)
        canvas.create_window(200, 220, window=button3)
        # canvas.create_window(200, 290, window=button4)

    def run_program1(self):
        self.run_program("Final\main.py")

    def run_program2(self):
        self.run_program("Final\\training.py")

    # def run_program3(self):
    #     self.run_program("recognise.py")
    
    def run_program3(self):
        self.run_program("Final\\adaptation.py")

    def run_program(self, program_name):
        if os.path.isfile(program_name):
            subprocess.Popen(["python", program_name], shell=True)
        else:
            print(f"Error: {program_name} not found.")

if __name__ == "__main__":
    root = tk.Tk()
    app = SafeFaceApp(root)
    root.mainloop()
