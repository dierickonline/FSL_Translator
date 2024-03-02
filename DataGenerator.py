import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import tkinter as tk

# Settings
padding = 20
imgSize = 300
counter = 0
folder = "Data/A"

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# win = tk.Tk()
# greeting = tk.Label(text="Hello, Tkinter")
# greeting.pack()


while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    imgFinal = np.ones((imgSize, imgSize, 3), np.uint8) * 255

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgCrop = img[y-padding:y+h+padding, x-padding:x+w+padding]

        ratio = h/w

        try:
            if ratio > 1:
                k = imgSize / h
                wCal = math.ceil(w * k)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize-wCal)/2)
                imgFinal[:, wGap:wGap+wCal] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(h * k)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgFinal[hGap:hGap + hCal, :] = imgResize

            cv2.imshow('ImageCrop', imgCrop)
            cv2.imshow('ImageFinal', imgFinal)
        except:
            print('Error')

    cv2.imshow("Image", img)

    key = cv2.waitKey(1)

    # When key S is pushed the image is saved to the folder
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgFinal)
        print(counter)

    elif key == ord("q"):
        exit()
