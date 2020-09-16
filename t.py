import cv2
import numpy as np

def showimage():
    imgfile = r'D:\1.png'
    img = cv2.imread(imgfile,cv2.IMREAD_COLOR)

    print(img)
    print(img.shape)
    # cv2.imshow('model', img)
    # cv2.waitKey(0)

showimage()