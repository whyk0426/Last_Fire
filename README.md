from picamera2 import Picamera2, Preview, libcamera
import cv2
import numpy as np

#from flask import Flask, Response
#app = Flask(__name__)

import serial
import time
ser = serial.Serial('/dev/ttyS0', 115200)
last_send_time = 0


picam = Picamera2()

config = picam.create_preview_configuration(main={"size":(1280, 720)}) #1280, 720
config["transform"] = libcamera.Transform(hflip=0, vflip=0)
picam.configure(config)

picam.start()


lower_fire = np.array([50, 100, 100]) #[50, 150, 150]
upper_fire = np.array([100, 255, 255]) #[100, 255, 255]


min_area = 400
max_area = 5000


line_color = (0, 0, 255)
circle_color = (0, 0, 0)


while True:
    img = picam.capture_array()
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mask = cv2.inRange(img, lower_fire, upper_fire)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    fire_detected = False
    
    center_x, center_y = img.shape[1] // 2, img.shape[0] // 2
    cross_length = 50
    
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        if min_area <= area <= max_area:
            fire_detected = True
            
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(img, center, radius, (255, 255, 255), 2)
            cv2.line(img, (int(x - 20) , int(y)), (int(x + 20), int(y)), (255, 255, 255), 2)
            cv2.line(img, (int(x), int(y - 20)), (int(x), int(y + 20)), (255, 255, 255), 2)
           
            distance = ((center_x - x)**2 + (center_y - y)**2)**0.5
              
            
            if distance <= 150:
                
                circle_color = (0, 0, 255) #Red
                line_color = (0, 0, 255) #Red
               
            else:
               
                circle_color = (0, 255, 255) #Yellow
                line_color = (255, 0, 0) #Blue
            
               
            cv2.circle(img, (img.shape[1] // 2, img.shape[0] // 2), 150, circle_color, 2)
            
#________________________(Serial Communication)__________________________________________________________________________            

            distance_x = (center_x - x) // 1 # fire is on left : distance_x > 0 , right : distance_x < 0
            distance_y = (center_y - y) // 1 
            
            current_time = time.time()
            if current_time - last_send_time >= 1:
                print(f"distance_x, distance_y value: {distance_x},{distance_y}")
                ser.write(str(distance_x).encode() + b',' + str(distance_y).encode())
                last_send_time = current_time
#________________________________________________________________________________________________________________________
        
                
    if not fire_detected:
        line_color = (255, 0, 0) #Blue
   
    cv2.line(img, (center_x - cross_length, center_y), (center_x + cross_length, center_y), line_color, 2)
    cv2.line(img, (center_x, center_y - cross_length), (center_x, center_y + cross_length), line_color, 2)

    cv2.imshow("Camera Preview", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        

picam.stop()
cv2.destroyAllWindows()
