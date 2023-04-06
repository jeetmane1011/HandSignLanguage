import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time


folder = 'Data/2'

counter = 0
offset = 30
imgSize = 300

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)

# drop image and display centered on whitescreen
def crop_img(x, y, w, h):
    imgWhite = np.ones((imgSize, imgSize, 3), dtype=np.uint8) * 255
    imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

    aspectRatio = h / w
    if aspectRatio > 1:
        k = imgSize / h
        wCal = math.ceil(k * w)
        imgResize = cv2.resize(imgCrop, (wCal, imgSize))
        wGap = math.ceil((imgSize - wCal) / 2)
        imgWhite[:, wGap:wCal + wGap] = imgResize

    else:
        k = imgSize / w
        hCal = math.ceil(k * h)
        imgResize = cv2.resize(imgCrop, (imgSize, hCal))
        hGap = math.ceil((imgSize - hCal) / 2)
        imgWhite[hGap:hCal + hGap, :] = imgResize

    return imgWhite


while True:
    success, img = cap.read()
    hands, img= detector.findHands(img)
    if hands:
        hand1 = hands[0]
        lm1 = hand1["lmList"]
        x1,y1,w1,h1 = hand1["bbox"]

        if len(hands) == 1:
            imgWhite = crop_img(x1, y1, w1, h1)
            # cv2.rectangle(img, (x1-offset, y1-offset), (x1+w1+offset, y1+h1+offset), (255, 0, 255), 5)

        elif len(hands) == 2:
            hand2 = hands[1]
            lm2 = hand2["lmList"]
            dist, info = detector.findDistance(lm1[0][0:2], lm2[0][0:2])

            if dist < 310:
                x2,y2,w2,h2 = hand2["bbox"]

                bbox_start = min(x1, x2), min(y1, y2)
                bbox_end = max(x1+w1, x2+w2), max(y1+h1, y2+h2)
                X, Y, W, H = bbox_start[0], bbox_start[1], bbox_end[0] - bbox_start[0], bbox_end[1] - bbox_start[1]

                imgWhite = crop_img(X, Y, W, H)
                # cv2.rectangle(img, (X-offset, Y-offset), (X+W+offset, Y+H+offset), (255, 0, 255), 5)
        
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('s'):
        counter+=1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)
