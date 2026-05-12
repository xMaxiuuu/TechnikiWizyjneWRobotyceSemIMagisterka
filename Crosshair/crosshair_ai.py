import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# 1. Ustawienia ścieżek
script_dir = Path(__file__).parent
folder_path = script_dir / 'base'
output_ai_path = script_dir / 'results_ai'

# Automatyczne tworzenie folderu na wyniki AI
output_ai_path.mkdir(exist_ok=True)

# 2. Ładowanie modelu (PyTorch pod spodem)
# 'yolov8n.pt' to lekki model, idealny do szybkich testów lokalnych
model = YOLO('yolov8n.pt') 

def process_ai_images(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Nie znaleziono: {image_path}")
        return

    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2

    # 3. Predykcja AI
    # Wyłączamy verbose=False, żeby nie zaśmiecać konsoli logami przy każdym zdjęciu
    results = model(img, verbose=False)[0]

    # Sprawdzamy czy cokolwiek wykryto
    if len(results.boxes) > 0:
        # Bierzemy pierwszy wykryty obiekt (najwyższa pewność)
        # xyxy to współrzędne ramki (Bounding Box)
        box = results.boxes[0].xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, box)

        # Obliczamy środek obiektu na podstawie ramki od AI
        obj_x = (x1 + x2) // 2
        obj_y = (y1 + y2) // 2

        dx = obj_x - cx
        dy = obj_y - cy

        # --- TWORZENIE ZDJĘCIA 1: DETEKCJA ---
        img_ann = img.copy()
        # Żółta ramka AI (Bounding Box)
        cv2.rectangle(img_ann, (x1, y1), (x2, y2), (0, 255, 255), 4)
        # Czerwony celownik (środek obrazu)
        cv2.drawMarker(img_ann, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 100, 5)
        # Linia do obiektu
        cv2.line(img_ann, (cx, cy), (obj_x, obj_y), (0, 0, 255), 3)
        # Napis z deltą
        text = f"AI Delta: X={dx} Y={dy}"
        cv2.putText(img_ann, text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 255), 5)

        # --- TWORZENIE ZDJĘCIA 2: PRZESUNIĘCIE ---
        T = np.float32([[1, 0, -dx], [0, 1, -dy]])
        centered_img = cv2.warpAffine(img, T, (w, h))
        # Zielony celownik potwierdzający wycentrowanie
        cv2.drawMarker(centered_img, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 100, 5)

        # --- ZAPIS W FOLDERZE results_ai ---
        orig_name = f"ai_detect_{image_path.stem}.jpg"
        centered_name = f"ai_centered_{image_path.stem}.jpg"
        
        cv2.imwrite(str(output_ai_path / orig_name), img_ann)
        cv2.imwrite(str(output_ai_path / centered_name), centered_img)

        print(f"Przetworzono {image_path.name}: Delta X: {dx}, Delta Y: {dy}")

        # Podgląd na ekranie
        preview = np.hstack((cv2.resize(img_ann, (640, 480)), 
                             cv2.resize(centered_img, (640, 480))))
        cv2.imshow("AI Result (Orig vs Centered)", preview)
        cv2.waitKey(800) # Automatyczne przejście po 0.8 sekundy

# Lista plików do testu
photos = ['photo1.jpeg', 'photo2.jpeg', 'photo3.jpeg', 'photo4.jpeg']

for p in photos:
    process_ai_images(folder_path / p)

cv2.destroyAllWindows()
print(f"\nSukces! Wszystkie zdjęcia AI znajdziesz w: {output_ai_path}")