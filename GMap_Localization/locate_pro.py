import cv2
import numpy as np
import os
import webbrowser
from PIL import Image
from PIL.ExifTags import GPSTAGS

# Konfiguracja
BASE_DIR = 'base'

def get_gps(path):
    """Bezpośredni odczyt GPS z metadanych zdjęcia."""
    try:
        with Image.open(path) as img:
            exif = img._getexif()
            if not exif: return None
            gps = {GPSTAGS.get(t, t): v for t, v in exif.get(34853, {}).items()}
            conv = lambda v: float(v[0] + v[1]/60.0 + v[2]/3600.0)
            lat, lon = conv(gps['GPSLatitude']), conv(gps['GPSLongitude'])
            if gps.get('GPSLatitudeRef') == 'S': lat = -lat
            if gps.get('GPSLongitudeRef') == 'W': lon = -lon
            return lat, lon
    except: return None

def get_dna(path):
    """Generowanie wizualnego odcisku palca (Barwa + Geometria)."""
    img = cv2.imread(path)
    if img is None: return None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    color = cv2.normalize(hist, hist).flatten()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edge = np.sum(cv2.Canny(gray, 100, 200) > 0) / gray.size
    return (color, edge)

# --- START ---
query = input("📍 PRZECIĄGNIJ ZDJĘCIE: ").strip().replace("'", "").replace('"', "")

# 1. Próba bezpośrednia
coords = get_gps(query)
if coords:
    print(f"✅ Znaleziono GPS w metadanych: {coords}")
else:
    print("🛰️ Brak metadanych. Szukam wizualnie w bazie...")
    q_dna = get_dna(query)
    best_match, max_score = None, -1
    
    for f in os.listdir(BASE_DIR):
        p = os.path.join(BASE_DIR, f)
        ref_gps = get_gps(p)
        ref_dna = get_dna(p)
        if ref_gps and ref_dna:
            score = (cv2.compareHist(q_dna[0], ref_dna[0], 0) * 0.7) + ((1 - abs(q_dna[1] - ref_dna[1])) * 0.3)
            if score > max_score:
                max_score, best_match, coords = score, f, ref_gps
    
    print(f"🎯 Dopasowano do: {best_match} ({round(max_score*100, 2)}%)")

# 2. Wynik i Mapa
if coords:
    lat, lon = coords
    print(f"📍 LOKALIZACJA: {lat}, {lon}")
    print(f"🔗 MAPA: https://www.google.com/maps/search/?api=1&query={lat},{lon}")
    print(f"🔗 WIDOK: https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={lat},{lon}")
    webbrowser.open(f"https://www.google.com/maps/search/?api=1&query={lat},{lon}")
    webbrowser.open(f"https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={lat},{lon}")
else:
    print("❌ Nie udało się ustalić pozycji.")