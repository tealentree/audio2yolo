import subprocess
import sys
import os
import time

def run_script(script_name):
    """
    Wykonuje podany skrypt Python jako osobny proces. 
    Jeśli skrypt zakończy się błędem, cały pipeline zostanie bezpiecznie zatrzymany.
    """
    print(f"\n{'='*55}")
    print(f"[STARTING STAGE] {script_name}")
    print(f"{'='*55}\n")
    
    try:
        script_path = os.path.join("scripts", script_name)
        subprocess.run([sys.executable, script_path], check=True)
        
        print(f"\n[STAGE COMPLETED] {script_name}")
        
    except subprocess.CalledProcessError as e:
        print(f"\n[CRITICAL ERROR] Script '{script_name}' crashed (exit code: {e.returncode}).")
        print("Halting the pipeline to prevent the generation of corrupted or incomplete data.")
        sys.exit(1)
        
    except FileNotFoundError:
        print(f"\n[FILE NOT FOUND] Cannot locate script: '{script_path}'.")
        print("Note: Ensure you are running 3_run_pipeline.py from the ROOT directory of the project.")
        sys.exit(1)

if __name__ == "__main__":
    print("Initializing automated Dataset Generation Pipeline...")
    start_time = time.time()
    
    # Lista skryptow do wykonania w kolejnosci
    execution_order = [
        "1_generate_audio.py",
        "2_generate_spectrograms.py"
    ]
    
    for script in execution_order:
        run_script(script)
        time.sleep(1) 
        
    end_time = time.time()
    total_duration = round(end_time - start_time, 2)
    
    print(f"\n{'='*55}")
    print(f"[PIPELINE SUCCESS] All stages completed successfully.")
    print(f"Total processing time: {total_duration} seconds.")
    print(f"{'='*55}")
    print("Your complete image dataset is now available in the '3_spectrograms' directory.")