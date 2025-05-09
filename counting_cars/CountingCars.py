import sys
sys.path.append('D:\Programs & tools\cmder')

import cvzone
from ultralytics import YOLO
import cv2
import math
import numpy as np
from sort import *

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("03-car counters/cars.mp4")
model = YOLO("yolo11n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

mask = cv2.imread("03-car counters/mask.png")

tracker = Sort(max_age = 20, min_hits = 3, iou_threshold = 0.3)

limits = [0, 360, 620, 360]

totalCount = []

while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)
    imgGraphics = cv2.imread("03-car counters/graphics.png", cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, imgGraphics, (0, 0))
    results = model(imgRegion, stream=True)
    detections = np.empty((0, 5))
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass == "car" or currentClass == "bus" or currentClass == "truck" and conf > 0.4:
                # cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)),
                #                scale = 0.6, thickness = 1, offset=3)
                # cvzone.cornerRect(img, (x1, y1, w, h), l=7, rt=5)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    trackerResults = tracker.update(detections)
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

    for result in trackerResults:
        x1, y1, x2, y2 , id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=5, rt=1, colorR=(255, 0, 0))
        cvzone.putTextRect(img, f'{int(id)} {classNames[cls]}', (x1, y1-5), scale = 0.6, thickness = 1, offset=3)

        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 3, (255, 0, 0), cv2.FILLED)

        if limits[0] < cx < limits[2] and limits[1] - 20 < cy < limits[3] + 20:
            if totalCount.count(id) == 0:
                totalCount.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

    #cvzone.putTextRect(img, f' Car Counter:{len(totalCount)}', (50, 50))
    cv2.putText(img, str(len(totalCount)), (300, 100), cv2.FONT_HERSHEY_TRIPLEX, 3, (0, 0, 255), 7)

    cv2.imshow("Webcam", img)
    #cv2.imshow("Mask", imgRegion)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    if key == ord('p'):
        cv2.waitKey(-1)
    # if cv2.waitKey(1) & 0xff == ord('q'):
    #     break


cap.release()
cv2.destroyAllWindows()
