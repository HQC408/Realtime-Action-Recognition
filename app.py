import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import subprocess
import signal

class App:
    def __init__(self, root):
        self.root = root
        self.process = None
        self.is_running = False

        self.setup_ui()

    def setup_ui(self):
        self.root.title("Action Recognition Test App")
        self.root.geometry("400x260")
        self.root.configure(bg='#F0F0F0')

        # Header
        header_frame = tk.Frame(self.root, bg='#2C3E50', height=40)
        header_frame.pack(fill=tk.X)
        header_label = tk.Label(header_frame, text="Action Recognition Test App", font=("Segoe UI", 14, "bold"), bg='#2C3E50', fg='white')
        header_label.pack(pady=5)

        # Main content
        content_frame = tk.Frame(self.root, bg='#F0F0F0')
        content_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

        # Data type selection
        data_type_label = tk.Label(content_frame, text="Select Data Type:", font=("Segoe UI", 10), bg='#F0F0F0')
        data_type_label.grid(row=0, column=0, pady=(5, 5), sticky='w')
        self.data_type_var = tk.StringVar(value="webcam")
        data_type_combo = ttk.Combobox(content_frame, textvariable=self.data_type_var, values=["video", "folder", "webcam"], state="readonly", font=("Segoe UI", 10))
        data_type_combo.grid(row=0, column=1, pady=(5, 5), padx=5)

        # Data path input
        data_path_label = tk.Label(content_frame, text="Data Path:", font=("Segoe UI", 10), bg='#F0F0F0')
        data_path_label.grid(row=1, column=0, pady=(5, 5), sticky='w')
        data_path_frame = tk.Frame(content_frame, bg='#F0F0F0')
        data_path_frame.grid(row=1, column=1, pady=(5, 5), padx=5)
        self.data_path_entry = tk.Entry(data_path_frame, width=30, font=("Segoe UI", 10), fg='grey')
        self.data_path_entry.insert(0, "If using camera, enter 0")  # Placeholder text
        self.data_path_entry.bind("<FocusIn>", self.clear_placeholder)
        self.data_path_entry.bind("<FocusOut>", self.add_placeholder)
        self.data_path_entry.pack(side=tk.LEFT, padx=(0, 5))
        browse_button = tk.Button(data_path_frame, text="Browse", command=self.browse_data_path, font=("Segoe UI", 9), bg='#5DADE2', fg='white')
        browse_button.pack(side=tk.LEFT)

        # Run and Stop buttons
        button_frame = tk.Frame(self.root, bg='#F0F0F0')
        button_frame.pack(pady=10)
        run_button = tk.Button(button_frame, text="Run", command=self.run_test, bg='#458e33', fg='black', font=("Segoe UI", 10, "bold"), width=10)
        run_button.pack(side=tk.LEFT, padx=5)
        stop_button = tk.Button(button_frame, text="Stop", command=self.stop_test, bg='#a63c4e', fg='black', font=("Segoe UI", 10, "bold"), width=10)
        stop_button.pack(side=tk.LEFT, padx=5)

        # Footer
        footer_frame = tk.Frame(self.root, bg='#2C3E50', height=25)
        footer_frame.pack(fill=tk.X, side=tk.BOTTOM)
        footer_label = tk.Label(footer_frame, text="CuongVip", font=("Segoe UI", 8), bg='#2C3E50', fg='white')
        footer_label.pack(pady=5)

    def clear_placeholder(self, event):
        if self.data_path_entry.get() == "If using camera, enter 0":
            self.data_path_entry.delete(0, tk.END)
            self.data_path_entry.config(fg='black')

    def add_placeholder(self, event):
        if self.data_path_entry.get() == "":
            self.data_path_entry.insert(0, "If using camera, enter 0")
            self.data_path_entry.config(fg='grey')

    def run_test(self):
        if self.is_running:
            messagebox.showwarning("Warning", "A test is already running.")
            return

        data_type = self.data_type_var.get()
        data_path = self.data_path_entry.get()

        if not data_path or data_path == "If using camera, enter 0":
            messagebox.showerror("Error", "Please fill in all fields.")
            return

        keyword = "Realtime-Action-Recognition-master"
        index = data_path.find(keyword)
        if index != -1:
            data_path_corrected = data_path[index + len(keyword) + 1:] 
        else:
            data_path_corrected = os.path.normpath(data_path)

        cmd = f'python src/s5_test.py --data_type {data_type} --data_path "{data_path_corrected}" --output_folder "output"'

        self.is_running = True
        self.process_thread = threading.Thread(target=self.run_command, args=(cmd,))
        self.process_thread.start()

    def run_command(self, cmd):
        # Sử dụng CREATE_NEW_PROCESS_GROUP trên Windows
        self.process = subprocess.Popen(cmd, shell=True, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        self.process.communicate()
        self.is_running = False

    def stop_test(self):
        if self.is_running and self.process:
            self.process.send_signal(signal.CTRL_BREAK_EVENT)  # Dùng CTRL_BREAK_EVENT trên Windows để ngắt tiến trình
            self.is_running = False
            messagebox.showinfo("Stopped", "The test has been stopped.")
        else:
            messagebox.showwarning("Warning", "No test is running.")

    def browse_data_path(self):
        data_type = self.data_type_var.get()
        file_path = ""
        if data_type == "video":
            file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.avi;*.mp4"), ("All files", "*.*")])
            if file_path and not file_path.lower().endswith(('.avi', '.mp4')):
                messagebox.showerror("Error", "Invalid video file format. Please select an .avi or .mp4 file.")
                return
        elif data_type == "folder":
            file_path = filedialog.askdirectory()
        self.data_path_entry.delete(0, tk.END)
        self.data_path_entry.insert(0, file_path)

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
