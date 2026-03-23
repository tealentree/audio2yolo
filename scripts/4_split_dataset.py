import os
import random
import shutil
import glob

SPLIT_RATIO = 0.20

IMAGES_TRAIN_DIR = "4_yolo_dataset/images/train"
IMAGES_TEST_DIR = "4_yolo_dataset/images/val"
LABELS_TRAIN_DIR = "4_yolo_dataset/labels/train"
LABELS_TEST_DIR = "4_yolo_dataset/labels/val"

def copy_files(image_paths, dest_images_dir, dest_labels_dir):
  for img_path in image_paths:
    base_filename = os.path.basename(img_path)
    filename_without_ext = os.path.splitext(base_filename)[0]

    txt_path = os.path.splitext(img_path)[0] + ".txt"
    
    shutil.copy(img_path, os.path.join(dest_images_dir, base_filename))

    dest_txt_path = os.path.join(dest_labels_dir, filename_without_ext + ".txt")
    if os.path.exists(txt_path):
      shutil.copy(txt_path, dest_txt_path)
    else:
      with open(dest_txt_path, 'w') as f:
        pass

def split_dataset():
  print(f"Splitting dataset (Train: {100 - SPLIT_RATIO * 100}%, Test: {SPLIT_RATIO * 100}%)")

  os.makedirs(IMAGES_TEST_DIR, exist_ok=True)
  os.makedirs(LABELS_TEST_DIR, exist_ok=True)

  for folder in [IMAGES_TRAIN_DIR, IMAGES_TEST_DIR, LABELS_TRAIN_DIR, LABELS_TEST_DIR]:
    os.makedirs(folder, exist_ok=True)

  all_images = glob.glob("3_spectrograms/*/*.png", recursive=True)
  print(f"Total images found: {len(all_images)}")

  if not all_images:
    print("No images found in '3_spectrograms'. Please run the previous stages of the pipeline first.")
    return
  
  random.shuffle(all_images)

  split_index = int(len(all_images) * (1 - SPLIT_RATIO))
  train_images = all_images[:split_index]
  test_images = all_images[split_index:]
  print(f"Training set: {len(train_images)} images")
  print(f"Testing set: {len(test_images)} images")
  print("Kopiowanie plików do folderu Training...")
  copy_files(train_images, IMAGES_TRAIN_DIR, LABELS_TRAIN_DIR)

  print("Kopiowanie plików do folderu Testing...")
  copy_files(test_images, IMAGES_TEST_DIR, LABELS_TEST_DIR)

  print("\n[SUKCES] Dane zostały podzielone, a labely sparowane. YOLO jest gotowe do treningu!")



if __name__ == "__main__":
  split_dataset()