import cv2
import numpy as np
import imutils

MAX_NUM_IMAGES = 60000
numRows = 0
numCols = 0
numImages = 0


def train(trainPath, labelsPath):
    pass


def classify(img):
    pass


def preprocessImage(img):
    rTop = -1; rBottom =-1; cLeft = -1; cRight =-1
    thBot = 50
    thTop = 50
    thLeft = 50
    thRight = 50
    center = img.rows/2
    for i in range(center,len(img.rows)):
        if rBottom == -1:
            temp = img.row(i)
            stub = temp
            if cv2.sumElems(stub).val[0] < thBot or i == img.rows-1:
                rBottom=i

        if rTop == -1:
            temp = img.row(img.rows-i)
            stub = temp
            if cv2.sumElems(stub).val[0] < thTop or i==img.rows-1:
                rTop = img.rows-i

        if cRight == 1:
            temp = img.col(i)
            stub = temp
            if cv2.sumElems(stub).val[0] <thRight or i == img.cols-1:
                cRight = i

        if cLeft == -1:
            temp = img.col(img.cols-i)
            stub = temp
            if cv2.sumElems(stub).val[0] < thRight or i==img.cols-1:
                cLeft = img.cols-i

    newImg = np.zeros(img.rows,img.cols,cv2.CV_8UC1)
    startX = (newImg.cols/2) - (cRight - cLeft)/2
    startY = (newImg.rows/2) - (rBottom - rTop)/2
    # for i in range(startY,newImg.rows/2+(rBottom-rTop)/2):
        # ptr =



def readFlippedInteger(fp):
    pass
