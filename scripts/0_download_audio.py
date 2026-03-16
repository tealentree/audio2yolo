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

def parse_input():
    parser = argparse.ArgumentParser()

    parser.add_argument("--url", required=True)
    parser.add_argument("--class", dest="folder", required=True) #class jest slowem kluczowym, wiec nalezy przekierowac pod inna nazwe

    arguments = parser.parse_args()

    url = arguments.url
    folder = arguments.folder

    if(folder not in AVAILABLE_CLASSES):
        print(f"BŁĄD: Wpisana klasa nie istnieje")
        sys.exit(1)
    return url, folder

url, folder = parse_input()

download_path = os.path.join(DOWNLOAD_FOLDER, folder)

os.makedirs(download_path, exist_ok=True)

download_options = {
    'format': 'bestaudio/best',
    'outtmpl': f'{download_path}/%(title)s.%(ext)s',
    # 'ffmpeg_location': './scripts/ffmpeg',

    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'wav',
        'preferredquality': '192',
    }],

    'postprocessor_args': [
        '-ar', '44100',
        '-ac', '1'
    ]

}


try:
    with yt_dlp.YoutubeDL(download_options) as ydl:
        ydl.download([url])
except yt_dlp.utils.DownloadError:
    print("Nie udało się pobrać filmu")
    sys.exit(1)
