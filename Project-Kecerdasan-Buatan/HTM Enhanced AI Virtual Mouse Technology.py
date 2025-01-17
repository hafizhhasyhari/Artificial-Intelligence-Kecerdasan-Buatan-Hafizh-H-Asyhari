// HTM Enhanced AI Virtual Mouse Technology //

# Mengimpor library yang diperlukan
import cv2  # Untuk pemrosesan gambar
import mediapipe as mp  # Untuk deteksi tangan
import time  # Untuk pengukuran waktu
import math  # Untuk perhitungan matematika
import numpy as np  # Untuk operasi array

# Kelas untuk mendeteksi tangan
class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        # Inisialisasi parameter deteksi tangan
        self.mode = mode  # Mode deteksi
        self.maxHands = maxHands  # Jumlah maksimum tangan yang akan dideteksi
        self.detectionCon = detectionCon  # Confidence threshold untuk deteksi
        self.trackCon = trackCon  # Confidence threshold untuk tracking

        # Inisialisasi MediaPipe Hands
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils  # Untuk menggambar landmark
        self.tipIds = [4, 8, 12, 16, 20]  # ID untuk ujung jari

    def findHands(self, img, draw=True):
        # Mengubah gambar dari BGR ke RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Memproses gambar untuk mendeteksi tangan
        self.results = self.hands.process(imgRGB)

        # Jika ada landmark tangan yang terdeteksi
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    # Menggambar landmark tangan pada gambar
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)

        return img  # Mengembalikan gambar dengan landmark

    def findPosition(self, img, handNo=0, draw=True):
        # Menyimpan posisi x dan y dari landmark
        xList = []
        yList = []
        bbox = []  # Bounding box
        self.lmList = []  # List untuk menyimpan posisi landmark
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]  # Mengambil tangan yang dipilih
            for id, lm in enumerate(myHand.landmark):
                # Mengambil tinggi, lebar, dan channel gambar
                h, w, c = img.shape
                # Menghitung posisi x dan y dari landmark
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)  # Menyimpan posisi x
                yList.append(cy)  # Menyimpan posisi y
                self.lmList.append([id, cx, cy])  # Menyimpan id dan posisi landmark
                if draw:
                    # Menggambar lingkaran pada posisi landmark
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            # Menghitung bounding box
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                # Menggambar bounding box pada gambar
                cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
                              (0, 255, 0), 2)

        return self.lmList, bbox  # Mengembalikan list landmark dan bounding box

    def fingersUp(self):
        fingers = []  # List untuk menyimpan status jari
        # Memeriksa jari jempol
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)  # Jempol terangkat
        else:
            fingers.append(0)  # Jempol tidak terangkat

        # Memeriksa jari-jari lainnya
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)  # Jari terangkat
            else:
                fingers.append(0)  # Jari tidak terangkat

        return fingers  # Mengembalikan status jari

    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
        # Mengambil posisi x dan y dari dua landmark yang diberikan
        x1, y1 = self.lmList[p1][1:]  # Koordinat landmark pertama
        x2, y2 = self.lmList[p2][1:]  # Koordinat landmark kedua
        # Menghitung titik tengah antara dua landmark
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            # Menggambar garis antara dua landmark
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            # Menggambar lingkaran pada posisi landmark pertama
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            # Menggambar lingkaran pada posisi landmark kedua
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            # Menggambar lingkaran pada titik tengah
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)

        # Menghitung jarak antara dua landmark menggunakan rumus Pythagoras
        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]  # Mengembalikan panjang, gambar, dan koordinat

# Fungsi utama untuk menjalankan program
def main():
    pTime = 0  # Waktu sebelumnya
    cTime = 0  # Waktu saat ini
    cap = cv2.VideoCapture(1)  # Mengambil video dari kamera
    detector = handDetector()  # Membuat objek handDetector

    while True:
        success, img = cap.read()  # Membaca frame dari kamera
        img = detector.findHands(img)  # Mendeteksi tangan dalam gambar
        lmList, bbox = detector.findPosition(img)  # Mencari posisi landmark dan bounding box
        if len(lmList) != 0:  # Jika ada landmark yang terdeteksi
            print(lmList[4])  # Mencetak posisi landmark jari telunjuk

        cTime = time.time()  # Mengambil waktu saat ini
        fps = 1 / (cTime - pTime)  # Menghitung frame per second (FPS)
        pTime = cTime  # Memperbarui waktu sebelumnya

        # Menampilkan FPS pada gambar
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)

        cv2.imshow("Image", img)  # Menampilkan gambar
        cv2.waitKey(1)  # Menunggu 1 ms untuk frame berikutnya

# Memastikan bahwa fungsi main() dijalankan saat file ini dieksekusi
if __name__ == "__main__":
    main()  # Menjalankan fungsi utama

# Watermark
# Oleh : Hafizh HIlman (hafizhhasyhari)
