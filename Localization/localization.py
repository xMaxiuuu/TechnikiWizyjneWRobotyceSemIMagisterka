import cv2
import numpy as np
import os

class CorridorLocator:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        # Parametry SIFT zoptymalizowane pod Twoje zdjęcia
        self.sift = cv2.SIFT_create(nfeatures=3000, contrastThreshold=0.03)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        self.database = []
        self.ref_names = []
        self.ref_imgs = []

    def prepare_image(self, img):
        """Konwersja do szarości i poprawa lokalnego kontrastu."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return self.clahe.apply(gray)

    def load_database(self):
        print("📦 Budowanie bazy korytarza (1-31)...")
        for i in range(1, 32):
            filename = f"{i}.jpeg"
            path = os.path.join(self.folder_path, filename)
            img = cv2.imread(path)
            if img is None: continue
            
            processed = self.prepare_image(img)
            kp, des = self.sift.detectAndCompute(processed, None)
            
            if des is not None:
                self.database.append((kp, des))
                self.ref_names.append(filename)
                self.ref_imgs.append(img)
        print(f"✅ Baza gotowa ({len(self.database)} zdjęć).\n")

    def locate(self, query_name):
        path = os.path.join(self.folder_path, query_name)
        query_img = cv2.imread(path)
        if query_img is None: return None
        
        q_proc = self.prepare_image(query_img)
        kp_q, des_q = self.sift.detectAndCompute(q_proc, None)
        
        best_idx = -1
        max_inliers = 0
        final_confidence = 0

        matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))

        for i, (kp_ref, des_ref) in enumerate(self.database):
            matches = matcher.knnMatch(des_q, des_ref, k=2)
            good = [m for m, n in matches if m.distance < 0.7 * n.distance]

            if len(good) > 5:
                src_pts = np.float32([kp_q[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_ref[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                
                _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if mask is not None:
                    inliers = np.sum(mask)
                    confidence = (inliers / len(good)) * 100
                    
                    if inliers > max_inliers:
                        max_inliers = inliers
                        best_idx = i
                        final_confidence = confidence

        if best_idx != -1:
            # Tworzenie wizualnego porównania do pliku
            h1, w1 = query_img.shape[:2]
            scale = 600 / h1
            img1_res = cv2.resize(query_img, (int(w1*scale), 600))
            img2_res = cv2.resize(self.ref_imgs[best_idx], (int(self.ref_imgs[best_idx].shape[1]*scale), 600))
            side_by_side = cv2.hconcat([img1_res, img2_res])
            
            label = f"R: {query_name} -> {self.ref_names[best_idx]} ({final_confidence:.1f}%)"
            cv2.putText(side_by_side, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imwrite(f"Comp_{query_name}", side_by_side)
            
            return {
                "ref": self.ref_names[best_idx],
                "conf": final_confidence
            }
        return None

# --- URUCHOMIENIE ---
FOLDER = "Base"
TESTY = ["R1.jpeg", "R2.jpeg", "R3.jpeg", "R4.jpeg", "R5.jpeg"]

locator = CorridorLocator(FOLDER)
locator.load_database()

print(f"{'ZDJĘCIE':<10} | {'WYNIK':<10} | {'PEWNOŚĆ %'}")
print("-" * 35)

for r_img in TESTY:
    result = locator.locate(r_img)
    if result:
        print(f"{r_img:<10} | {result['ref']:<10} | {result['conf']:.1f}%")
    else:
        print(f"{r_img:<10} | Brak dopasowania")