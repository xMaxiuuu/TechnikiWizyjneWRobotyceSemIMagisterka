import cv2
import numpy as np

def rozpoznaj_i_podlicz(sciezka_do_obrazu):
    img = cv2.imread(sciezka_do_obrazu)
    if img is None:
        print("Nie znaleziono pliku obrazu!")
        return

    # 1. STANDARYZACJA ROZMIARU
    wysokosc, szerokosc = img.shape[:2]
    docelowa_szerokosc = 800
    skala = docelowa_szerokosc / szerokosc
    img = cv2.resize(img, (docelowa_szerokosc, int(wysokosc * skala)))
    output = img.copy()
    
    # Kopia obrazu w palecie HSV (idealna do sprawdzania kolorów)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 2. PRZETWARZANIE
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Zmniejszyłem rozmycie (z 17 na 13), aby nie usuwało krawędzi brakujących monet
    blurred = cv2.medianBlur(gray, 13)

    # 3. DETEKCJA OKRĘGÓW (bardziej czuła)
    circles = cv2.HoughCircles(
        blurred, 
        cv2.HOUGH_GRADIENT, 
        dp=1.2,            
        minDist=35,        
        param1=45,         
        param2=30,         # Niższy próg by złapać wszystkie monety
        minRadius=14,      
        maxRadius=55       
    )

    # 4. BAZA MONET Z KOLORAMI ŚRODKA
    monety_pl = [
        {"val": 5.0,  "d": 24.0, "name": "5zl",  "srodek": "srebrny"}, # Bimetal: srebrny środek
        {"val": 1.0,  "d": 22.7, "name": "1zl",  "srodek": "srebrny"},
        {"val": 2.0,  "d": 21.5, "name": "2zl",  "srodek": "zloty"},   # Bimetal: złoty środek
        {"val": 0.5,  "d": 20.5, "name": "50gr", "srodek": "srebrny"},
        {"val": 0.05, "d": 19.5, "name": "5gr",  "srodek": "zloty"},
        {"val": 0.2,  "d": 18.5, "name": "20gr", "srodek": "srebrny"},
        {"val": 0.02, "d": 17.5, "name": "2gr",  "srodek": "zloty"},
        {"val": 0.1,  "d": 16.5, "name": "10gr", "srodek": "srebrny"},
        {"val": 0.01, "d": 15.5, "name": "1gr",  "srodek": "zloty"}
    ]

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        # Sortujemy od największej (dla kalibracji skali)
        circles = sorted(circles, key=lambda x: x[2], reverse=True)

        max_r_px = circles[0][2]
        skala_px_na_mm = (max_r_px * 2) / 24.0 

        suma_calkowita = 0.0

        for i, (x, y, r) in enumerate(circles):
            srednica_px = r * 2
            srednica_mm = srednica_px / skala_px_na_mm
            
            # --- ROZPOZNAWANIE KOLORU ---
            # Tworzymy maskę wycinającą sam środek monety (promień * 0.5)
            mask = np.zeros(img.shape[:2], dtype="uint8")
            cv2.circle(mask, (x, y), int(r * 0.5), 255, -1)
            
            # Obliczamy średnią wartość kolorów wewnątrz środka
            mean_hsv = cv2.mean(hsv_img, mask=mask)
            saturacja = mean_hsv[1]
            
            # Jeśli saturacja jest niska, to szarość/srebro. Jeśli wysoka - złoto/miedź.
            if saturacja > 40: 
                wykryty_srodek = "zloty"
                kolor_ramki = (0, 215, 255)  # Rysuje na żółto
            else:
                wykryty_srodek = "srebrny"
                kolor_ramki = (192, 192, 192) # Rysuje na szaro/srebrno
            
            # Filtrujemy bazę: sprawdzamy wymiary TYLKO wśród monet o odpowiednim kolorze!
            kandydaci = [m for m in monety_pl if m["srodek"] == wykryty_srodek]
            match = min(kandydaci, key=lambda m: abs(m["d"] - srednica_mm))
            
            # Odchylenie rzędu 3.5 mm dopuszczamy (perspektywa/odblaski)
            if abs(match["d"] - srednica_mm) > 3.5:
                continue

            suma_calkowita += match["val"]
            
            # Rysowanie kółek w kolorze wykrytego materiału
            cv2.circle(output, (x, y), r, kolor_ramki, 2)
            cv2.rectangle(output, (x-25, y-15), (x+35, y+5), (0,0,0), -1)
            cv2.putText(output, match['name'], (x-20, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, kolor_ramki, 1)

        print(f"Wykryto {len(circles)} monet.")
        print(f"Łączna kwota: {suma_calkowita:.2f} PLN")
        
        cv2.putText(output, f"SUMA: {suma_calkowita:.2f} PLN", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        cv2.imshow("Wynik Rozpoznawania z Kolorem", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Nie wykryto monet.")

rozpoznaj_i_podlicz('monety.jpg')