import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import warnings

warnings.filterwarnings('ignore')

def generate_mel_spectrogram(audio_path, output_path, sr=44100, n_mels=128):
    """
    Konwertuje plik audio na Mel-spektrogram i zapisuje go jako obraz PNG.
    """
    try:
        y, sr = librosa.load(audio_path, sr=sr)
        
        # Generacja Mel-spektogramu
        mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=n_mels)
        
        # Konwersja do skali decybelowej (logarytmicznej) dla lepszej wizualizacji
        mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)
        
        plt.figure(figsize=(10, 4), frameon=False)
        plt.axis('off')
        
        librosa.display.specshow(mel_spect_db, sr=sr, hop_length=512, x_axis='time', y_axis='mel', cmap='magma')

        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, transparent=True)
        
        plt.close()
        
    except Exception as e:
        print(f"Error processing file {audio_path}: {e}")

def process_all_classes(base_input_folder, base_output_folder):
    if not os.path.exists(base_input_folder):
        print(f"Error: Input directory '{base_input_folder}' does not exist.")
        return

    class_subfolders = [f for f in os.listdir(base_input_folder) 
                        if os.path.isdir(os.path.join(base_input_folder, f))]

    if not class_subfolders:
        print(f"Warning: No subdirectories found in '{base_input_folder}'.")
        return

    print(f"Found {len(class_subfolders)} classes to process: {class_subfolders}")

    for class_name in class_subfolders:
        class_input_path = os.path.join(base_input_folder, class_name)
        class_output_path = os.path.join(base_output_folder, class_name)
        
        os.makedirs(class_output_path, exist_ok=True)
        
        wav_files = glob.glob(os.path.join(class_input_path, "*.wav"))
        file_count = len(wav_files)
        
        if file_count == 0:
            print(f"Skipping class '{class_name}': No .wav files found.")
            continue
            
        print(f"\nProcessing class '{class_name}' ({file_count} files)...")
        
        for index, audio_path in enumerate(wav_files, start=1):
            file_name = os.path.basename(audio_path)
            png_name = os.path.splitext(file_name)[0] + ".png"
            output_path = os.path.join(class_output_path, png_name)
            
            generate_mel_spectrogram(audio_path, output_path)
            print(f"  [{index}/{file_count}] Saved: {png_name}")

if __name__ == "__main__":
    INPUT_FOLDER = "2_processed_audio" 
    OUTPUT_FOLDER = "3_spectrograms"

    print("Starting spectrogram generation pipeline...")
    process_all_classes(INPUT_FOLDER, OUTPUT_FOLDER)
    print("\nProcess completed successfully. Image dataset is ready for labeling.")