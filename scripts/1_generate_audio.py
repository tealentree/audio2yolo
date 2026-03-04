import os
import random
import numpy as np
import librosa
import soundfile as sf
import glob
import warnings
from scipy.signal import butter, filtfilt

# Ignoruj warningi z librosy
warnings.filterwarnings('ignore')

# --- 1. KONFIGURACJA ---
INPUT_FOLDER = "1_raw_audio/targets"
DRONE_BG_FILE = "1_raw_audio/backgrounds/drone-sound.wav"
OUTPUT_FOLDER = "2_processed_audio"

SAMPLE_DURATION_SEC = 30.0
SAMPLE_RATE = 44100
VERSIONS_PER_CLASS = 3

# Definicja klas: typy i wymaganie ilości eventów dla dźwięków przerywanych
AUDIO_CLASSES = {
    "woda": {"type": "ciagly"},
    "pozar": {"type": "ciagly"},
    "kolumna": {"type": "ciagly"},
    "krab": {"type": "przerywany", "count": 5},
    "bomba": {"type": "przerywany", "count": 5},
    "karabin": {"type": "przerywany", "count": 3}
}

def bandpass_filter(audio, sr, low_freq=150, high_freq=10000):
    """
    Filtr pasmowo-przepustowy, przepuszczajacy tylko dzwieki o okreslonych czestotliwosciach.
    Symuluje glosnik
    """
    nyq = 0.5 * sr
    low = low_freq / nyq
    high = high_freq / nyq
    b, a = butter(4, [low, high], btype='band')

    return filtfilt(b, a, audio)

def prepare_drone_background(total_samples):
    """
    Laduje i zapetla dźwięk tla drona, aby dopasowac go do docelowej dlugosci probki. 
    Zwieksza glosnosc bazowego dzwieku o okreslony mnoznik.
    """
    try:
        drone_noise, _ = librosa.load(DRONE_BG_FILE, sr=SAMPLE_RATE)
        background_canvas = np.copy(drone_noise)
        
        while len(background_canvas) < total_samples:
            background_canvas = np.concatenate((background_canvas, background_canvas))
        
        # Przytnij do dokładnie 30 sekund
        start_time = 2.0
        start_time = int(start_time * SAMPLE_RATE)
        background_canvas = background_canvas[start_time:start_time + total_samples]
        
        # Zwiększ głośność tła, aby było bardziej wyraźne w miksie (mnożnik można dostosować)
        background_canvas = background_canvas * 1.5 
        return background_canvas
        
    except Exception as e:
        print(f"Error loading drone background noise: {e}")
        print(f"Please ensure the file exists at: {DRONE_BG_FILE}")
        return None

def load_audio_pool(folder_path):
    """
    Laduje wszystkie pliki .wav z podanego folderu i zwraca liste tablic audio.
    """
    wav_files = glob.glob(os.path.join(folder_path, "*.wav"))
    loaded_audio = []
    
    for file_path in wav_files:
        try:
            audio_data, _ = librosa.load(file_path, sr=SAMPLE_RATE)
            audio_data = bandpass_filter(audio_data, SAMPLE_RATE)
            loaded_audio.append(audio_data)
        except Exception as e:
            print(f"Failed to load {file_path}: {e}")
            
    return loaded_audio


def generate_dataset():
    """
    Główna funkcja generująca dataset. Dla każdej klasy dźwiękowej tworzy określoną liczbę wariantów.
    """
    total_samples = int(SAMPLE_DURATION_SEC * SAMPLE_RATE)
    
    # Bazowe tło drona, które będzie używane we wszystkich wariantach
    base_background = prepare_drone_background(total_samples)
    if base_background is None:
        return

    for class_name, config in AUDIO_CLASSES.items():
        source_folder = os.path.join(INPUT_FOLDER, class_name)
        target_folder = os.path.join(OUTPUT_FOLDER, class_name)
        
        # Załaduj wszystkie dostępne próbki dźwiękowe dla tej klasy
        audio_pool = load_audio_pool(source_folder)
        
        if not audio_pool:
            print(f"Skipping class '{class_name}': No .wav files found in {source_folder}")
            continue

        os.makedirs(target_folder, exist_ok=True)
        print(f"\nProcessing class: {class_name.upper()} (Type: {config['type']}, Sources found: {len(audio_pool)})")
        
        file_id = 0
        for audio_id in range(len(audio_pool)):
            for version_idx in range(VERSIONS_PER_CLASS):
                canvas = np.copy(base_background)
                
                # --- LOGIKA DZWIEKOW CIAGLYCH ---
                if config["type"] == "ciagly":
                    target_audio = audio_pool[audio_id]
                    continuous_audio = np.copy(target_audio)

                    while len(continuous_audio) < total_samples:
                        continuous_audio = np.concatenate((continuous_audio, continuous_audio))
                    
                    continuous_audio = continuous_audio[:total_samples]
                    
                    # Modyfikacja glosnosci dzwieku aby symulowac rozna odleglosc od zrodla
                    event_volume = random.uniform(0.1, 1)
                    canvas += (continuous_audio * event_volume)

                # --- LOGIKA DZWIEKOW PRZERYWANYCH ---
                elif config["type"] == "przerywany":
                    event_count = config["count"]
                    segment_size = total_samples // event_count

                    for event_idx in range(event_count):
                        target_audio = audio_pool[audio_id]
                        audio_length = len(target_audio)
                        
                        event_volume = random.uniform(0.1, 1.0)
                        modified_audio = target_audio * event_volume
                        
                        margin = segment_size - audio_length
                        segment_start_idx = margin // 2 if margin > 0 else 0
                            
                        absolute_start_idx = (event_idx * segment_size) + segment_start_idx
                        absolute_end_idx = absolute_start_idx + audio_length
                        
                        if absolute_end_idx > total_samples:
                            available_space = total_samples - absolute_start_idx
                            canvas[absolute_start_idx:total_samples] += modified_audio[:available_space]
                        else:
                            canvas[absolute_start_idx:absolute_end_idx] += modified_audio

                max_amplitude = np.max(np.abs(canvas))
                if max_amplitude > 1.0:
                    canvas = canvas / max_amplitude 

                output_filename = f"{class_name}_version_{file_id}_{event_volume:.2f}.wav"
                output_filepath = os.path.join(target_folder, output_filename)
                
                sf.write(output_filepath, canvas, SAMPLE_RATE)
                print(f"  [Saved] {output_filename}")
                file_id += 1

if __name__ == "__main__":
    print("Starting audio processing pipeline: 1_raw_audio -> 2_processed_audio")
    generate_dataset()
    print("\nDataset generation completed successfully.")