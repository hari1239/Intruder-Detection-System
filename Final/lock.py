import tkinter as tk
from tkinter import messagebox

class LockingSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("Locking System")

        # Set the password
        self.correct_password = "harini"  # Replace with your desired password

        # Create widgets
        self.label = tk.Label(root, text="Enter Password:")
        self.password_entry = tk.Entry(root, show="*")
        self.unlock_button = tk.Button(root, text="Unlock", command=self.check_password)

        # Grid layout
        self.label.grid(row=0, column=0, pady=10)
        self.password_entry.grid(row=0, column=1, pady=10)
        self.unlock_button.grid(row=1, column=0, columnspan=2)

    def check_password(self):
        entered_password = self.password_entry.get()

        if entered_password == self.correct_password:
            messagebox.showinfo("Success", "Password is correct. System unlocked!")
            # Add your unlocking logic or open the main application window here
            self.root.destroy()  # Close the locking system window
        else:
            messagebox.showerror("Error", "Incorrect password. Please try again.")

if __name__ == "__main__":
    root = tk.Tk()
    app = LockingSystem(root)
    root.mainloop()
