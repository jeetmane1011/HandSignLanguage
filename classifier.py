import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

counter = 0
offset = 30
imgSize = 300

model_folder = 'e35'

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
classifier = Classifier(f'{model_folder}/keras_model.h5', f'{model_folder}/labels.txt')

labels = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',]

word = ""
old_text = ""
count_same_ltr = 0

# crop, resize image and make centered on whitescreen
def crop_img(x, y, w, h, frame):
    imgWhite = np.ones((imgSize, imgSize, 3), dtype=np.uint8) * 255
    imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

    cv2.rectangle(frame, (x - offset, y - offset), (x + w+ offset, y + h + offset), (0, 255, 0), 4)

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

    prediction, index = classifier.getPrediction(imgWhite, draw=False)
    return imgWhite, labels[index]


while True:
    success, img = cap.read()
    imgOutput = img.copy()
    blackboard = np.zeros((480, 480, 3), dtype=np.uint8)
    hands, img = detector.findHands(img)
    if hands:
        hand1 = hands[0]
        lm1 = hand1["lmList"]
        x1,y1,w1,h1 = hand1["bbox"]
        if len(hands) == 1:
            imgWhite, text = crop_img(x1, y1, w1, h1, imgOutput)

        elif len(hands) == 2:
            hand2 = hands[1]
            lm2 = hand2["lmList"]
            dist, info = detector.findDistance(lm1[0][0:2], lm2[0][0:2])

            if dist < 310:
                x2,y2,w2,h2 = hand2["bbox"]

                bbox_start = min(x1, x2), min(y1, y2)
                bbox_end = max(x1+w1, x2+w2), max(y1+h1, y2+h2)
                X, Y, W, H = bbox_start[0], bbox_start[1], bbox_end[0] - bbox_start[0], bbox_end[1] - bbox_start[1]

                imgWhite, text = crop_img(X, Y, W, H, imgOutput)
        
        if old_text == text:
            count_same_ltr += 1
        else:
            count_same_ltr = 0
            old_text = text
        
        if count_same_ltr > 20:
            word = word + text
            count_same_ltr = 0

        
        cv2.putText(blackboard, "Predicted Character- " + text, (30, 100), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 0))
        
        cv2.imshow("ImageWhite", imgWhite)
    else:
        text = " "
        if old_text != text:
            word = word + " "
            old_text = " "

    # wrap text within board

    cv2.putText(blackboard, word, (30, 240), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 255, 255))
    frame = np.hstack((imgOutput, blackboard))
    cv2.imshow("Recognizing gesture", frame)
    cv2.waitKey(1)

