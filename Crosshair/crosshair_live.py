import cv2
import numpy as np
from pathlib import Path

# --- KONFIGURACJA ---
W, H = 1200, 800 # Stały rozmiar okna dla MacBooka
script_dir = Path(__file__).parent
folder_path = script_dir / 'base'

# 1. Wybór zdjęcia
print("Wybierz zdjęcie (1, 2, 3 lub 4):")
photo_num = input("Numer: ")
img_path = folder_path / f'photo{photo_num}.jpeg'

img_raw = cv2.imread(str(img_path))
if img_raw is None:
    exit(f"Błąd: Nie znaleziono pliku {img_path}")

# Skalujemy obraz do stałego okna
img_view = cv2.resize(img_raw, (W, H))

# 2. DETEKCJA CELU (Zamiast trenowania AI)
# Szukamy czarnego obiektu na jasnym tle
gray = cv2.cvtColor(img_view, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

target_x, target_y = W // 2, H // 2 # Domyślnie środek
if contours:
    c = max(contours, key=cv2.contourArea)
    M = cv2.moments(c)
    if M["m00"] != 0:
        target_x = int(M["m10"] / M["m00"])
        target_y = int(M["m01"] / M["m00"])

# 3. INTERFEJS I MYSZKA
# Startujemy celownikiem na środku okna
mouse_x, mouse_y = W // 2, H // 2

def on_mouse(event, x, y, flags, param):
    global mouse_x, mouse_y
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = x, y

cv2.namedWindow("Nawigator Precyzyjny")
cv2.setMouseCallback("Nawigator Precyzyjny", on_mouse)

print(f"Uruchomiono. Cel wykryty na: {target_x}, {target_y}")

while True:
    canvas = img_view.copy()
    
    # Logika instrukcji (Porównanie kursora z wykrytym celem)
    dx = target_x - mouse_x
    dy = target_y - mouse_y
    
    instr = []
    # Czułość 30 pikseli
    if dx > 30: instr.append("PRAWO")
    elif dx < -30: instr.append("LEWO")
    if dy > 30: instr.append("DOL")
    elif dy < -30: instr.append("GORA")

    # Wyświetlanie
    if not instr:
        msg, color = "NAMIERZONO!", (0, 255, 0)
        cv2.circle(canvas, (mouse_x, mouse_y), 40, color, 2)
    else:
        msg, color = f"NAWIGACJA: {' '.join(instr)}", (0, 0, 255)
        # Linia od kursora do czarnego przedmiotu
        cv2.line(canvas, (mouse_x, mouse_y), (target_x, target_y), (255, 255, 0), 1)

    cv2.putText(canvas, msg, (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    cv2.drawMarker(canvas, (mouse_x, mouse_y), (255, 255, 255), cv2.MARKER_CROSS, 30, 2)

    cv2.imshow("Nawigator Precyzyjny", canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()