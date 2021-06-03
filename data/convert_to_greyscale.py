#!/usr/bin/python3
import cv2
import os

# convert all jpg/png images in the current folder from RGB to grayscale color profile.

if __name__ == "__main__":
    files = os.listdir(".")
    for file in files:
        if file.endswith(".jpg") or file.endswith(".png"):
            image = cv2.imread(file, cv2.IMREAD_UNCHANGED)
            converted = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            cv2.imwrite(file, converted)
