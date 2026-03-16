# Moduł Detekcji Akustycznej (UAV)

Projekt automatycznego generowania i przetwarzania datasetu audio (Dźwięki ciągłe i przerywane) dla drona, przygotowujący dane do uczenia modelu YOLO.

## Jak używać tego repozytorium?

1. Sklonuj repozytorium na swój komputer.
2. Zbuduj początkową strukturę folderów dla surowych plików wrzucając ten kod w terminal (lub stwórz je ręcznie):
   `mkdir -p 1_raw_audio/backgrounds`
   `mkdir -p 1_raw_audio/targets/{woda,pozar,kolumna,krab,bomba,karabin}`
3. Do folderu `1_raw_audio/backgrounds/` wrzuć plik `drone-sound.wav`.
4. Do odpowiednich folderów w `1_raw_audio/targets/` wrzuć czyste nagrania źródłowe z dysku google lub pobierz za pomocą skryptu 0_download_audio.py:
   * Do poprawnego zadziałania skryptu potrzebne jest pobranie FFmpeg do konwersji audio:
      * Wejdź na stronę: `https://www.gyan.dev/ffmpeg/builds/`
      * Pobierz plik: `ffmpeg-release-essentials.zip`
      * W folderze `scripts/ffmpeg` umieść plik ffmpeg.exe
   * Aby odpowiednio pobrać scieżkę audio należy podać:
      * **--url**: Poprawny link do filmu z pożądaną ścieżką dźwiękową
      * **--class**: Klasa, którą reprezentuje wybrana ścieżka audio
   * Przykladowe uzycie: `python scripts/0_download_audio.py --url "https://youtube.com/watch?v=https://www.youtube.com/watch?v=rzSAXlHLsL8" --class krab`
5. Uruchom główny pipeline:
   `python scripts/3_run_pipeline.py`

Skrypt automatycznie wygeneruje 30-sekundowe zaszumione miksy (w folderze `2_processed_audio`), a następnie zamieni je na gotowe do treningu Mel-spektrogramy (w folderze `3_spectrograms`).
