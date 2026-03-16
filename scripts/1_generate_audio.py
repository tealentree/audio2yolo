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
LABELS_OUTPUT_FOLDER = "3_spectrograms"

SAMPLE_DURATION_SEC = 30.0
SAMPLE_RATE = 44100
VERSIONS_PER_CLASS = 2

# Definicja klas: typy i wymaganie ilości eventów dla dźwięków przerywanych
AUDIO_CLASSES = {
    "woda": {"type": "ciagly"},
    "pozar": {"type": "ciagly"},
    "kolumna": {"type": "ciagly"},
    "krab": {"type": "przerywany", "count": 5},
    "bomba": {"type": "przerywany", "count": 5},
    "karabin": {"type": "przerywany", "count": 3}
}

CLASS_TO_ID = {
    "woda": 0,
    "pozar": 1,
    "kolumna": 2,
    "krab": 3,
    "bomba": 4,
    "karabin": 5
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

def slice_into_windows(audio, window_size_sec=3.0, step_size_sec=1.0):
    """
    Dodaje efekt przesuwającego się okna dźwiękowego.
    Okno jest nakładane na audio, aby stworzyć naturalny efekt odbierania dźwięku w czasie rzeczywistym.
    """
    audio_length = len(audio)
    window_size_samples = int(window_size_sec * SAMPLE_RATE)
    step_size_samples = int(step_size_sec * SAMPLE_RATE)

    sliced_windows = []
    timestamps = []

    for start in range(0, audio_length, step_size_samples):
        end = min(start + window_size_samples, audio_length)
        sliced_windows.append(audio[start:end]) # Lista z pocietymi dzwiekami

        timestamps.append(start / SAMPLE_RATE) # Lista z czasami poczatkow okien (w sekundach)
    return sliced_windows, timestamps


def calculate_yolo_bbox(window_start, window_end, event_start, event_end):
    """
    Oblicza współrzędne bounding boxa w formacie YOLO dla danego okna i zdarzenia.
    Zwraca (x_center, y_center, width, height) znormalizowane do zakresu [0, 1].
    """
    window_duration = window_end - window_start
    
    if event_end <= window_start or event_start >= window_end:
        return None
    
    visible_start = max(window_start, event_start)
    visible_end = min(window_end, event_end)

    rel_start = (visible_start - window_start) / window_duration
    rel_end = (visible_end - window_start) / window_duration

    x_center = (rel_start + rel_end) / 2.0
    width = rel_end - rel_start
    
    y_center = 0.5
    height = 1.0

    if width < 0.05:
        return None
    
    return x_center, y_center, width, height


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
        labels_target_folder = os.path.join(LABELS_OUTPUT_FOLDER, class_name)

        # Załaduj wszystkie dostępne próbki dźwiękowe dla tej klasy
        audio_pool = load_audio_pool(source_folder)
        
        if not audio_pool:
            print(f"Skipping class '{class_name}': No .wav files found in {source_folder}")
            continue

        os.makedirs(target_folder, exist_ok=True)
        os.makedirs(labels_target_folder, exist_ok=True)
        print(f"\nProcessing class: {class_name.upper()} (Type: {config['type']}, Sources found: {len(audio_pool)})")
        
        file_id = 0
        for audio_id in range(len(audio_pool)):
            for version_idx in range(VERSIONS_PER_CLASS):
                canvas = np.copy(base_background)
                current_file_volume = random.uniform(0.1, 0.7)

                event_registry = [] 

                # --- LOGIKA DZWIEKOW CIAGLYCH ---
                if config["type"] == "ciagly":
                    target_audio = audio_pool[audio_id]
                    continuous_audio = np.copy(target_audio)

                    while len(continuous_audio) < total_samples:
                        continuous_audio = np.concatenate((continuous_audio, continuous_audio))
                    
                    continuous_audio = continuous_audio[:total_samples]
                    canvas += (continuous_audio * current_file_volume)

                    # Rejestruje zdarzenia dla dźwięków ciągłych jako jedno zdarzenie trwające przez całą próbkę (format dla YOLO)
                    event_registry.append({
                        "class_id": CLASS_TO_ID[class_name],
                        "start_sample": 0,
                        "end_sample": total_samples
                    })
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

                        event_registry.append({
                            "class_id": CLASS_TO_ID[class_name],
                            "start_sample": absolute_start_idx,
                            "end_sample": absolute_end_idx
                        })

                max_amplitude = np.max(np.abs(canvas))
                if max_amplitude > 1.0:
                    canvas = canvas / max_amplitude 

                window_size_sec = 3.0
                output_audio, timestamps = slice_into_windows(canvas, window_size_sec)

                for idx, segment in enumerate(output_audio):
                    window_start_sec = timestamps[idx]
                    window_end_sec = window_start_sec + window_size_sec

                    base_filename = f"{class_name}_version_{file_id}_{idx}"

                    wav_filepath = os.path.join(target_folder, base_filename + ".wav")
                    sf.write(wav_filepath, segment, SAMPLE_RATE)

                    yolo_labels = []

                    for event in event_registry:
                        event_start_sec = event["start_sample"] / SAMPLE_RATE
                        event_end_sec = event["end_sample"] / SAMPLE_RATE
                        class_id = event["class_id"]

                        bbox = calculate_yolo_bbox(window_start_sec, window_end_sec, event_start_sec, event_end_sec)
                        if bbox is not None:
                            x_center, y_center, width, height = bbox
                            label_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                            yolo_labels.append(label_line)

                    txt_filepath = os.path.join(labels_target_folder, base_filename + ".txt")
                    with open(txt_filepath, 'w') as f:
                        f.writelines(yolo_labels)

                    print(f"[Saved] {base_filename}.wav and .txt ({len(yolo_labels)} events)")
                    
                file_id += 1


if __name__ == "__main__":
    print("Starting audio processing pipeline: 1_raw_audio -> 2_processed_audio")
    generate_dataset()
    print("\nDataset generation completed successfully.")