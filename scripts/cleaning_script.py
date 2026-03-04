import os
"""
Czysci wszystkie dane stworzone przez pipeline, NIE USUWA ZRODEL DZWIEKOW (1_raw_audio)
"""
def delete_files_in_folder(folder_path):
  if not os.path.exists(folder_path):
    print(f"Error: The folder '{folder_path}' does not exist.")
    return
  if not os.path.isdir(folder_path):
    print(f"Error: The path '{folder_path}' is not a directory.")
    return
  try:
    for filename in os.listdir(folder_path):
      file_path = os.path.join(folder_path, filename)
      if os.path.isfile(file_path):
        os.remove(file_path)
        print(f"Deleted: {file_path}")
    
    print(f"All files in '{folder_path}' have been deleted.")
  except Exception as e:
    print(f"An error occurred while deleting files: {e}")

def clean_all():
  for folder_name in ["2_processed_audio", "3_spectrograms"]:
    for subfolder_name in ["krab", "bomba", "woda", "pozar", "karabin", "kolumna"]:
      target_folder = os.path.join(folder_name, subfolder_name)
      delete_files_in_folder(target_folder)
  print("Cleaning completed for all target folders.")

if __name__ == "__main__":
  # target_folder = "2_processed_audio/karabin"  # Specify the target folder to clean
  # delete_files_in_folder(target_folder)
  clean_all()