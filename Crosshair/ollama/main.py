import cv2
import numpy as np
import ollama
import re
import os

def get_object_coordinates(image_path):
    """Ollama podaje nam przybliżony obszar obiektu."""
    prompt = (
        "Identify the small black rectangular device (marked 4K) on the wooden surface. "
        "Respond ONLY with the bounding box coordinates in exactly this format: "
        "[ymin, xmin, ymax, xmax] using normalized values (0.0 to 1.0)."
    )
    try:
        response = ollama.chat(
            model='llava',
            messages=[{'role': 'user', 'content': prompt, 'images': [image_path]}]
        )
        output_text = response['message']['content']
        print(f"-> Ollama widzi obiekt w okolicy: {output_text}")
        numbers = re.findall(r"0\.\d+|\.\d+|\d+\.\d+", output_text)
        if len(numbers) >= 4:
            return [float(n) for n in numbers[:4]]
        return None
    except Exception:
        return None

def process_and_center(image_path, output_path):
    if not os.path.exists(image_path):
        return

    # 1. Wczytanie obrazu
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    print(f"-> Przetwarzam: {image_path} ({w}x{h})")

    # 2. Pobieramy przybliżone współrzędne z Ollamy
    coords = get_object_coordinates(image_path)
    
    # 3. METODA KLASYCZNA (Z Twojego przykładu) - Szukamy czarnego przedmiotu precyzyjnie
    # Konwersja na szarość i progowanie (szukamy czarnego elementu)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    final_x, final_y = cx, cy # Domyślnie środek
    found_rect = None

    if contours:
        # Wybieramy największy czarny kontur
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M["m00"] != 0:
            # Precyzyjny środek geometryczny obiektu
            final_x = int(M["m10"] / M["m00"])
            final_y = int(M["m01"] / M["m00"])
            found_rect = cv2.boundingRect(c)
            print(f"-> Precyzyjnie namierzono obiekt na: X={final_x}, Y={final_y}")

    # 4. Rysowanie zielonej ramki na oryginalnym zdjęciu (wokół obiektu 4K)
    if found_rect:
        rx, ry, rw, rh = found_rect
        cv2.rectangle(img, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 4)

    # 5. Obliczanie przesunięcia do środka
    dx = final_x - cx
    dy = final_y - cy
    
    # Macierz przesunięcia
    T = np.float32([[1, 0, -dx], [0, 1, -dy]])
    centered_img = cv2.warpAffine(img, T, (w, h), borderValue=(0, 0, 0))

    # 6. RYSOWANIE BARDZO GRUBEGO CELOWNIKA NA ŚRODKU
    color_red = (0, 0, 255)
    thick = 10    # Jeszcze grubszy niż ostatnio
    size = 120    # Większy celownik
    
    # Krzyż
    cv2.line(centered_img, (cx - size, cy), (cx + size, cy), color_red, thick)
    cv2.line(centered_img, (cx, cy - size), (cx, cy + size), color_red, thick)
    # Kropka centralna
    cv2.circle(centered_img, (cx, cy), 15, color_red, -1)

    # Zapis
    cv2.imwrite(output_path, centered_img)
    print(f"-> Sukces! Obraz wycentrowany zapisany w: {output_path}\n")

if __name__ == "__main__":
    process_and_center("photo1.jpeg", "wynik_wycentrowany.jpg")