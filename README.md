from picamera2 import Picamera2, Preview, libcamera
import cv2
import numpy as np
from ultralytics import YOLO


import serial #serial communication
import time
ser = serial.Serial('/dev/ttyAMA0', 115200)
last_send_time = 0


picam = Picamera2()

config = picam.create_preview_configuration(main={"size": (1280, 720)}) 
config["transform"] = libcamera.Transform(hflip=0, vflip=0)
picam.configure(config)

picam.start()

model = YOLO("last.pt")


import os #saving video
save_path = "saved_videos"  
fourcc = cv2.VideoWriter_fourcc(*'XVID')  
video_filename = os.path.join(save_path, f'video_{time.time()}.avi')  
video_writer = cv2.VideoWriter(video_filename, fourcc, 2.0, (1280, 720))  

while True:
    img = picam.capture_array()
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    results = model(img, show=False, conf=0.5)  #yolov8
    fire_detected = False

    
    for result in results:
        boxes = result.boxes
        for box in boxes: #box boundary
            x1, y1, x2, y2 = box.xyxy[0]  
            conf = box.conf[0]  
            cls = int(box.cls[0]) 
            
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2) 
            cv2.putText(img, f'Fire Detected [{conf:.2f}]', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            if cls == 0:
                fire_detected = True #center of fire
                cx = (x1 + x2) // 2 
                cy = (y1 + y2) // 2
            
            distance_x = (cx - center_x) // 1 
            distance_y = (center_y - cy) // 1 
                
            current_time = time.time() #serial communication
            if current_time - last_send_time >= 1: 
                print(f"Fire is on: {distance_x},{distance_y}")
                ser.write(str(distance_x).encode() + b',' + str(distance_y).encode())
                last_send_time = current_time    
                
                
    center_x, center_y = img.shape[1] // 2, img.shape[0] // 2 # drawing cross
    cross_length = 50
    line_color = (0, 255, 255) if fire_detected else (255, 0, 0) 
    cv2.line(img, (center_x - cross_length, center_y), (center_x + cross_length, center_y), line_color, 2)
    cv2.line(img, (center_x, center_y - cross_length), (center_x, center_y + cross_length), line_color, 2)
    
    video_writer.write(img) #saving video
    
    cv2.imshow("Fire Detecting", img) #showing video

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

picam.stop()
video_writer.release()
cv2.destroyAllWindows()
