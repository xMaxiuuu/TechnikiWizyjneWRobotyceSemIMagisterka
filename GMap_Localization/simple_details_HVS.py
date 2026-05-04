import cv2
import numpy as np
import os
import webbrowser
import folium
import sys
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

# --- KONFIGURACJA ---
BASE_DIR = 'base'
MAP_FILE = 'mapa_bazy.html'

def get_gps_data(image_path):
    """Pobiera współrzędne ze zdjęcia referencyjnego."""
    try:
        with Image.open(image_path) as img:
            exif = img._getexif()
            if not exif: return None
            gps_info = {GPSTAGS.get(t, t): v for t, v in exif.get(34853, {}).items()}
            to_deg = lambda v: float(v[0] + v[1]/60.0 + v[2]/3600.0)
            lat = to_deg(gps_info['GPSLatitude'])
            lon = to_deg(gps_info['GPSLongitude'])
            if gps_info.get('GPSLatitudeRef') == 'S': lat = -lat
            if gps_info.get('GPSLongitudeRef') == 'W': lon = -lon
            return lat, lon
    except: return None

def extract_features(image_path):
    """Analiza wizualna: Barwy (HSV) i Detale (Krawędzie)."""
    img = cv2.imread(image_path)
    if img is None: return None, None
    
    # 1. Porównywanie barw (Histogram HSV)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    color_sig = cv2.normalize(hist, hist).flatten()
    
    # 2. Porównywanie krawędzi i detali (Canny)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / edges.size
    
    return color_sig, edge_density

# --- PROCES GŁÓWNY ---

print("="*60)
print(" 📍 GMap Localization 📍 ".center(60, "#"))
print("="*60)

# 1. Indeksowanie zdjęć
print("\n📸 [INDEKSOWANIE] Rozpoczynam skanowanie zasobów z folderu 'base'...")
database = []
mapa = folium.Map(location=[53.425, 14.519], zoom_start=16)

# 2. Zgrywanie metadanych i mapa
print("🗺️ [METADANE] Pobieranie danych EXIF oraz generowanie mapy punktów...")
for file in os.listdir(BASE_DIR):
    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
        path = os.path.join(BASE_DIR, file)
        gps = get_gps_data(path)
        color, edge = extract_features(path)
        
        if gps and color is not None:
            database.append({'path': path, 'gps': gps, 'color': color, 'edge': edge})
            folium.Marker(location=gps, popup=file, icon=folium.Icon(color='blue')).add_to(mapa)

mapa.save(MAP_FILE)
abs_map_path = os.path.abspath(MAP_FILE)
print(f" ✅ STATUS: Mapa bazy została pomyślnie wygenerowana.")
print(f" 📂 ŚCIEŻKA: {abs_map_path}")

# Pobranie ścieżki do zdjęcia testowego
print("\n" + "-"*60)
query_path = input(" 📥 PODAJ PEŁNĄ ŚCIEŻKĘ DO ZDJĘCIA TESTOWEGO: ").strip().replace("'", "").replace('"', "")
print("-"*60)

q_color, q_edge = extract_features(query_path)

if q_color is not None:
    best_match = None
    max_score = -1

    print("\n🧠 [ANALIZA] Porównywanie cech wizualnych obrazu...")
    
    for ref in database:
        # Porównywanie barw metodą korelacji
        sim_color = cv2.compareHist(q_color, ref['color'], cv2.HISTCMP_CORREL)
        
        # Porównywanie gęstości krawędzi
        sim_edge = 1.0 - abs(q_edge - ref['edge'])
        
        # Wynik końcowy (70% kolor, 30% krawędzie)
        total_score = (sim_color * 0.7) + (sim_edge * 0.3)
        
        if total_score > max_score:
            max_score = total_score
            best_match = ref
            res_color = sim_color
            res_edge = sim_edge

    if best_match:
        lat, lon = best_match['gps']
        
        print("\n" + " 📊 RAPORT ANALIZY 📊 ".center(60, "-"))
        print(f" > Najbliższe dopasowanie: {best_match['path']}")
        print(f" > Zgodność barw (HSV): {round(res_color * 100, 2)}%")
        print(f" > Zgodność detali (Canny): {round(res_edge * 100, 2)}%")
        print(f" > PRAWDOPODOBIEŃSTWO KOŃCOWE: {round(max_score * 100, 2)}%")
        
        pin_url = f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"
        sv_url = f"https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={lat},{lon}"
        
        print("\n" + " 🌍 LOKALIZACJA GEOGRAFICZNA 🌍 ".center(60, "-"))
        print(f" > Współrzędne geo: {lat}, {lon}")
        print(f" > Link do pinezki: {pin_url}")
        print(f" > Link do Street View: {sv_url}")
        
        # INTERAKCJA UŻYTKOWNIKA
        print("\n" + "="*60)
        choice = input(" 📍  SAM ENTER: Otwórz mapy | WPISZ 'Q' + ENTER: Zamknij program 📍  ").strip().lower()
        
        if choice == 'q':
            print("\n👋 [INFO] Zamykanie programu. Do zobaczenia!")
            os._exit(0)  # Natychmiastowe zabicie procesu
        
        print("\n🌐 [OTWIERANIE] Uruchamiam Mapy Google w przeglądarce...")
        webbrowser.open(pin_url)
        webbrowser.open(sv_url)
        print("="*60)
    else:
        print("\n❌ [BŁĄD] Nie znaleziono dopasowania w bazie danych.")
else:
    print("\n❌ [BŁĄD] Nie można przetworzyć pliku. Sprawdź ścieżkę.")