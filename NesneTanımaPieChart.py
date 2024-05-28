import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt


model = YOLO('yolov8n.pt')  


cap = cv2.VideoCapture(0)  


def calculate_distance(width_in_frame, real_width, focal_length):
    
    distance = (real_width * focal_length) / width_in_frame
    return distance


FOCAL_LENGTH = 700  
REAL_WIDTH = 0.5    


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


plt.ion()
fig = plt.figure(figsize=(10, 5))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    
    results = model(frame)
    
    object_info = []

    for result in results:
        for bbox in result.boxes:
            x1, y1, x2, y2 = bbox.xyxy[0].int().numpy()  # Nesne koordinatları
            label = model.names[int(bbox.cls)]  # Nesne sınıfı
            confidence = bbox.conf.item()  # Güven seviyesi

            
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

    
    if object_info:
        update_display(object_info)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()
