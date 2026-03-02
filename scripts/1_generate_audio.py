import os
import random
import numpy as np
import librosa
import soundfile as sf
import glob
import warnings

# Suppress librosa warnings for cleaner console output
warnings.filterwarnings('ignore')

# --- 1. CONFIGURATION ---
INPUT_FOLDER = "1_raw_audio/targets"
DRONE_BG_FILE = "1_raw_audio/backgrounds/drone-sound.wav"
OUTPUT_FOLDER = "2_processed_audio"

SAMPLE_DURATION_SEC = 30.0
SAMPLE_RATE = 44100
VERSIONS_PER_CLASS = 5

# Class definitions: types and required event counts
# Note: Folder names in '1_raw_audio/targets' must match these keys exactly.
AUDIO_CLASSES = {
    "woda": {"type": "continuous"},
    "pozar": {"type": "continuous"},
    "kolumna": {"type": "continuous"},
    "krab": {"type": "discrete", "count": 5},
    "bomba": {"type": "discrete", "count": 5},
    "karabin": {"type": "discrete", "count": 3}
}

def prepare_drone_background(total_samples):
    """
    Loads and loops the drone background noise to match the target sample duration.
    Increases the base volume by a specified multiplier.
    """
    try:
        drone_noise, _ = librosa.load(DRONE_BG_FILE, sr=SAMPLE_RATE)
        background_canvas = np.copy(drone_noise)
        
        # Loop the audio if it is shorter than the required duration
        while len(background_canvas) < total_samples:
            background_canvas = np.concatenate((background_canvas, background_canvas))
        
        # Trim to exact required length
        background_canvas = background_canvas[:total_samples]
        
        # Increase background volume
        background_canvas = background_canvas * 1.5 
        return background_canvas
        
    except Exception as e:
        print(f"Error loading drone background noise: {e}")
        print(f"Please ensure the file exists at: {DRONE_BG_FILE}")
        return None

def load_audio_pool(folder_path):
    """
    Loads all .wav files from a specified directory into memory.
    Returns a list of audio arrays.
    """
    wav_files = glob.glob(os.path.join(folder_path, "*.wav"))
    loaded_audio = []
    
    for file_path in wav_files:
        try:
            audio_data, _ = librosa.load(file_path, sr=SAMPLE_RATE)
            loaded_audio.append(audio_data)
        except Exception as e:
            print(f"Failed to load {file_path}: {e}")
            
    return loaded_audio

def generate_dataset():
    """
    Main pipeline function to generate mixed audio datasets based on class configurations.
    """
    total_samples = int(SAMPLE_DURATION_SEC * SAMPLE_RATE)
    
    # 1. Prepare the base drone background
    base_background = prepare_drone_background(total_samples)
    if base_background is None:
        return

    # 2. Iterate through each defined audio class
    for class_name, config in AUDIO_CLASSES.items():
        source_folder = os.path.join(INPUT_FOLDER, class_name)
        target_folder = os.path.join(OUTPUT_FOLDER, class_name)
        
        # Load all available source variations for the current class
        audio_pool = load_audio_pool(source_folder)
        
        if not audio_pool:
            print(f"Skipping class '{class_name}': No .wav files found in {source_folder}")
            continue

        os.makedirs(target_folder, exist_ok=True)
        print(f"\nProcessing class: {class_name.upper()} (Type: {config['type']}, Sources found: {len(audio_pool)})")
        
        for version_idx in range(VERSIONS_PER_CLASS):
            # Create a fresh copy of the background for this specific variation
            canvas = np.copy(base_background)
            
            # --- LOGIC FOR CONTINUOUS SOUNDS (Water, Fire, Convoy) ---
            if config["type"] == "continuous":
                target_audio = random.choice(audio_pool)
                continuous_audio = np.copy(target_audio)
                
                # Loop target audio to fill the 30-second duration
                while len(continuous_audio) < total_samples:
                    continuous_audio = np.concatenate((continuous_audio, continuous_audio))
                
                continuous_audio = continuous_audio[:total_samples]
                
                # Apply random volume modifier simulating distance
                volume_modifier = random.uniform(0.1, 0.7)
                canvas += (continuous_audio * volume_modifier)

            # --- LOGIC FOR DISCRETE SOUNDS (Krab, Bomb, Machine Gun) ---
            elif config["type"] == "discrete":
                event_count = config["count"]
                segment_size = total_samples // event_count
                
                for event_idx in range(event_count):
                    target_audio = random.choice(audio_pool)
                    audio_length = len(target_audio)
                    
                    event_volume = random.uniform(0.1, 0.7)
                    modified_audio = target_audio * event_volume
                    
                    # Calculate random start position within the current time segment
                    end_margin = segment_size - audio_length
                    segment_start_idx = random.randint(0, end_margin) if end_margin > 0 else 0
                        
                    absolute_start_idx = (event_idx * segment_size) + segment_start_idx
                    absolute_end_idx = absolute_start_idx + audio_length
                    
                    # Ensure the audio overlay doesn't exceed the total canvas length
                    if absolute_end_idx > total_samples:
                        available_space = total_samples - absolute_start_idx
                        canvas[absolute_start_idx:total_samples] += modified_audio[:available_space]
                    else:
                        canvas[absolute_start_idx:absolute_end_idx] += modified_audio

            # --- CLIPPING PREVENTION & EXPORT ---
            # Normalize the waveform if the mixed audio exceeds the digital maximum (1.0)
            max_amplitude = np.max(np.abs(canvas))
            if max_amplitude > 1.0:
                canvas = canvas / max_amplitude 

            output_filename = f"{class_name}_version_{version_idx + 1}.wav"
            output_filepath = os.path.join(target_folder, output_filename)
            
            sf.write(output_filepath, canvas, SAMPLE_RATE)
            print(f"  [Saved] {output_filename}")

if __name__ == "__main__":
    print("Starting audio processing pipeline: 1_raw_audio -> 2_processed_audio")
    generate_dataset()
    print("\nDataset generation completed successfully.")