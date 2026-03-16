from ultralytics import YOLO

# 1. Ładujemy TWÓJ wytrenowany model (podmień 'train' na odpowiedni folder, jeśli masz train2, train3)
model = YOLO("runs/detect/yolo_experiment/weights/best.pt")

# 2. Wybieramy obrazki do przetestowania (np. z naszego folderu walidacyjnego)
# Możesz tu podać ścieżkę do jednego pliku .png lub do całego folderu
source_to_test = "4_yolo_dataset/images/val"

# 3. Uruchamiamy detekcję
print("Rozpoczynam testowanie modelu...")
results = model.predict(
    source=source_to_test, 
    conf=0.5,      # Próg pewności: model pokaże ramkę tylko, jeśli jest na 50%+ pewny
    save=True,     # Zapisuje obrazki z narysowanymi ramkami
    show=False     # Zmień na True, jeśli chcesz, by obrazki od razu wyskakiwały na ekranie
)

print("\n[ZAKOŃCZONO] Sprawdź folder 'runs/detect/predict' aby zobaczyć wyniki!")