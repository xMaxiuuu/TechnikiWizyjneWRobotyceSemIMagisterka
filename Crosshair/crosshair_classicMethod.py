import cv2
import numpy as np
from pathlib import Path

# Ścieżki
script_dir = Path(__file__).parent
folder_path = script_dir / 'base'
output_path = script_dir / 'results'
output_path.mkdir(exist_ok=True) # Tworzy folder results, jeśli nie istnieje

file_names = ['photo1.jpeg', 'photo2.jpeg', 'photo3.jpeg', 'photo4.jpeg']

def process_and_save(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        return

    h, w = img.shape[:2]
    center_x, center_y = w // 2, h // 2

    # 1. Detekcja
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M["m00"] != 0:
            obj_x = int(M["m10"] / M["m00"])
            obj_y = int(M["m01"] / M["m00"])
            
            dx = obj_x - center_x
            dy = obj_y - center_y

            # --- ZDJĘCIE 1: ORYGINAŁ Z DOPISKIEM ---
            img_annotated = img.copy()
            # Celownik i linia
            cv2.drawMarker(img_annotated, (center_x, center_y), (0, 0, 255), cv2.MARKER_CROSS, 100, 5)
            cv2.line(img_annotated, (center_x, center_y), (obj_x, obj_y), (0, 0, 255), 3)
            
            # Tekst z informacją o przesunięciu
            info_text = f"Delta X: {dx}px, Delta Y: {dy}px"
            cv2.putText(img_annotated, info_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                        3, (0, 0, 255), 5, cv2.LINE_AA)

            # --- ZDJĘCIE 2: PRZESUNIĘTE (WYCENTROWANE) ---
            T = np.float32([[1, 0, -dx], [0, 1, -dy]])
            centered_img = cv2.warpAffine(img, T, (w, h))
            # Na tym zdjęciu pokazujemy tylko środek (celownik)
            cv2.drawMarker(centered_img, (center_x, center_y), (0, 255, 0), cv2.MARKER_CROSS, 100, 5)

            # --- ZAPIS I WYŚWIETLANIE ---
            # Zapisujemy pliki
            cv2.imwrite(str(output_path / f"result_{image_path.stem}_orig.jpg"), img_annotated)
            cv2.imwrite(str(output_path / f"result_{image_path.stem}_centered.jpg"), centered_img)

            # Podgląd (zmniejszony)
            combined = np.hstack((cv2.resize(img_annotated, (640, 480)), 
                                 cv2.resize(centered_img, (640, 480))))
            cv2.imshow(f"Wynik: {image_path.name}", combined)
            cv2.waitKey(1000) # Czeka 1 sekunda i idzie dalej automatycznie

# Start
for name in file_names:
    process_and_save(folder_path / name)

cv2.destroyAllWindows()
print(f"Gotowe! Wyniki zapisano w: {output_path}")