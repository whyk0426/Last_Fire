from picamera2 import Picamera2, Preview, libcamera
import cv2
import numpy as np
from ultralytics import YOLO

import serial
import time
import os  # 파일 경로 작업을 위해 추가
ser = serial.Serial('/dev/ttyAMA0', 115200)
last_send_time = 0

picam = Picamera2()

config = picam.create_preview_configuration(main={"size": (1280, 720)})  # 1280, 720
config["transform"] = libcamera.Transform(hflip=0, vflip=0)
picam.configure(config)

picam.start()

model = YOLO("last.pt")

# 저장할 이미지의 경로 지정
save_path = "saved_videos"  # 동영상 저장 폴더
if not os.path.exists(save_path):
    os.makedirs(save_path)  # 폴더가 없으면 생성

# 동영상 저장을 위한 VideoWriter 초기화
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 코덱 설정 (XVID 사용)
video_filename = os.path.join(save_path, f'video_{time.time()}.avi')  # 파일명 설정
video_writer = cv2.VideoWriter(video_filename, fourcc, 30.0, (1280, 720))  # FPS는 30으로 설정

while True:
    img = picam.capture_array()
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Run YOLOv8 inference on the frame
    results = model(img, show=False, conf=0.3)  # show=False로 설정하여 화면에 표시하지 않음
    
    fire_detected = False  # 불 감지 여부를 나타내는 플래그
    fire_center_coords = None  # 불의 중심 좌표를 저장할 새로운 변수

    # YOLO의 결과를 OpenCV 이미지에 그리기
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # 바운딩 박스 좌표
            conf = box.conf[0]  # 신뢰도
            cls = int(box.cls[0])  # 클래스 ID
            
            # 바운딩 박스 그리기
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # 초록색 박스
            cv2.putText(img, f'Class: {cls}, Conf: {conf:.2f}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # 만약 감지된 클래스가 '불'이라면
            if cls == 1:  # 여기서 1은 불 클래스의 ID라고 가정
                fire_detected = True  # 불이 감지되었음을 표시
                cx = (int(x1) + int(x2)) // 2  # 중심 x 좌표
                cy = (int(y1) + int(y2)) // 2  # 중심 y 좌표
                fire_center_coords = (cx, cy)  # 새로운 변수에 저장

    # Center cross
    center_x, center_y = img.shape[1] // 2, img.shape[0] // 2
    cross_length = 50

    # 십자가 색상 결정
    line_color = (0, 255, 255) if fire_detected else (255, 0, 0)  # 불이 감지되면 노란색, 아니면 파란색

    cv2.line(img, (center_x - cross_length, center_y), (center_x + cross_length, center_y), line_color, 2)
    cv2.line(img, (center_x, center_y - cross_length), (center_x, center_y + cross_length), line_color, 2)

    # 불의 중심 좌표가 계산되었을 경우 표시
    if fire_center_coords:
        cv2.circle(img, fire_center_coords, 10, (0, 255, 255), -1)  # 노란색 원으로 중심 표시

    # 동영상에 프레임 추가
    video_writer.write(img)  # 현재 프레임을 동영상 파일에 저장

    cv2.imshow("Camera Preview", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 동영상 저장 종료
video_writer.release()
picam.stop()
cv2.destroyAllWindows()
