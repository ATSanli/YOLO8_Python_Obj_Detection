import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

# YOLOv8 modelini yükle
model = YOLO('yolov8n.pt')  # Küçük model, daha büyük modeller için 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt' kullanabilirsiniz.

# Kameradan görüntü al
cap = cv2.VideoCapture(0)  # Kameradan görüntü almak için. Video dosyası için 'videofile.mp4' kullanabilirsiniz.

# Mesafe ölçme fonksiyonu (basit versiyon, kameranın kalibrasyon verilerine göre ayarlanmalı)
def calculate_distance(width_in_frame, real_width, focal_length):
    # Mesafe hesaplama formülü: (gerçek genişlik * odak uzaklığı) / çerçevedeki genişlik
    distance = (real_width * focal_length) / width_in_frame
    return distance

# Kameranın kalibrasyon verileri (örnek değerler)
FOCAL_LENGTH = 700  # Piksel cinsinden odak uzaklığı
REAL_WIDTH = 0.5    # Metre cinsinden nesnenin gerçek genişliği (örneğin, bir asker için)

# Pie chart ve liste güncelleme fonksiyonu
def update_display(object_info):
    plt.clf()
    distances = [info[1] for info in object_info]
    labels = [f'{info[0]}: {info[1]:.2f}m' for info in object_info]
    plt.subplot(121)
    plt.pie(distances, labels=labels, autopct='%1.1f%%')
    plt.title('Object Distances (meters)')
    
    plt.subplot(122)
    plt.axis('off')
    plt.title('Object List')
    for i, info in enumerate(object_info):
        plt.text(0, 1 - (i * 0.1), f'{i+1}. {info[0]}: {info[1]:.2f}m', fontsize=12)
    
    plt.draw()
    plt.pause(0.005)

# Matplotlib için interaktif mod
plt.ion()
fig = plt.figure(figsize=(10, 5))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO modelini kullanarak nesne tespiti yap
    results = model(frame)
    
    object_info = []

    for result in results:
        for bbox in result.boxes:
            x1, y1, x2, y2 = bbox.xyxy[0].int().numpy()  # Nesne koordinatları
            label = model.names[int(bbox.cls)]  # Nesne sınıfı
            confidence = bbox.conf.item()  # Güven seviyesi

            # Nesnenin merkezini ve genişliğini hesapla
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            width_in_frame = x2 - x1

            # Mesafeyi hesapla
            distance = calculate_distance(width_in_frame, REAL_WIDTH, FOCAL_LENGTH)
            object_info.append((label, distance))

            # Bilgileri ekrana yazdır
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f"Distance: {distance:.2f}m", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('YOLOv8 Object Detection', frame)

    # Pie chart ve liste güncelleme
    if object_info:
        update_display(object_info)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()