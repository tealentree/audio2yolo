import tkinter as tk
from tkinter import ttk, messagebox
import argparse
import yt_dlp
import os
import sys


DOWNLOAD_FOLDER = "1_raw_audio/targets"
AVAILABLE_CLASSES = ["bomba",
                     "karabin",
                     "kolumna",
                     "krab",
                     "pozar",
                     "woda"]

def normalize_time(t):
    parts = t.split(":")
    
    if len(parts) == 2:
        m, s = parts
        return f"00:{int(m):02}:{int(s):02}"
        
    if len(parts) == 3:
        h, m, s = parts
        return f"{int(h):02}:{int(m):02}:{int(s):02}"

    raise ValueError("Bad timestamp")

def download_clip():
    status_var.set("")
    root.update_idletasks()

    url = url_entry.get()
    folder = folder_entry.get()
    start = normalize_time(start_entry.get())
    end = normalize_time(end_entry.get())

    if(folder not in AVAILABLE_CLASSES):
        status_var.set("BŁĄD: Wpisana klasa nie istnieje")
        root.update_idletasks()
        return

    if end <= start:
        status_var.set("BŁĄD: Podany start klipu powinien byc szybciej niz podany koniec klipu")
        root.update_idletasks()
        return
    
    download_path = os.path.join(DOWNLOAD_FOLDER, folder)

    os.makedirs(download_path, exist_ok=True)

    download_options = {
        'format': 'bestaudio/best',
        'outtmpl': f'{download_path}/%(title)s.%(ext)s',
        'ffmpeg_location': './scripts/ffmpeg',

        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],

        'postprocessor_args': [
            '-ss', start,
            '-to', end,
            '-ar', '44100',
            '-ac', '1'
        ]

    }

    try:
        with yt_dlp.YoutubeDL(download_options) as ydl:
            ydl.download([url])
    except yt_dlp.utils.DownloadError:
        status_var.set("Pobieranie filmu zakonczone")
        root.update_idletasks()
        return
    
    status_var.set("Pobieranie filmu zakonczone")
    root.update_idletasks()
    

root = tk.Tk()
root.title("Audio Dataset Downloader")
root.geometry("500x400")

tk.Label(root, text="YouTube URL").pack()
url_entry = tk.Entry(root, width=60)
url_entry.pack()

tk.Label(root, text="Class").pack()
folder_entry = ttk.Combobox(root, values=AVAILABLE_CLASSES)
folder_entry.pack()

tk.Label(root, text="Start").pack()
start_entry = tk.Entry(root)
start_entry.pack()

tk.Label(root, text="End").pack()
end_entry = tk.Entry(root)
end_entry.pack()

tk.Button(root, text="Add Clip", command=download_clip).pack(pady=5)

status_var = tk.StringVar()
status_var.set("")

status_label = tk.Label(root, textvariable=status_var)
status_label.pack(pady=5)

root.mainloop()


