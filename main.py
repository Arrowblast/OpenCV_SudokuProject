import cv2
import numpy as np
from numpy import tan, cos, sin, pi, fabs, sqrt
import imutils
import math
import random
import sys

from matplotlib.pyplot import gray


def mergeRelatedLines(lines, img):
    # print(img.size)
    height, width, channels = img.shape
    # print(lines)
    for c in range(len(lines)):
        current = lines[c][0]
        # print(lines[current][0][0])
        if np.logical_and(current[0] == 0, current[1] == -100):
            continue
        p1 = current[0]
        theta1 = current[1]
        pt1current = []
        pt2current = []
        if (theta1 > pi * 45 / 180 and theta1 < pi * 135 / 180):
            pt1current = (0, p1 / sin(theta1))
            pt2current = (width, -width / tan(theta1) + p1 / sin(theta1))
        else:
            pt1current = (p1 / cos(theta1), 0)
            pt2current = (-height / tan(theta1) + p1 / cos(theta1), height)

        for p in range(len(lines)):
            pos = lines[p][0]
            if (current[0] == pos[0] and current[1] == pos[1]):
                continue
            if (fabs(pos[0] - current[0]) < 20 and fabs(pos[1] - current[1]) < pi * 10 / 180):
                p = pos[0]
                theta = pos[1]
                pt1 = []
                pt2 = []

                if (pos[1] > pi * 45 / 180 and pos[1] < pi * 135 / 180):
                    pt1 = (0, p / sin(theta))
                    pt2 = (width, -width / tan(theta) + p / sin(theta))
                else:
                    pt1 = (p / cos(theta), 0)
                    pt2 = (-height / tan(theta) + p / cos(theta), height)
                if (((pt1[0] - pt1current[0]) * (pt1[0] - pt1current[0]) + (pt1[1] - pt1current[1]) * (
                            pt1[1] - pt1current[1]) < 64 * 64) and
                        ((pt2[0] - pt2current[0]) * (pt2[0] - pt2current[0]) + (pt2[1] - pt2current[1]) * (
                                    pt2[1] - pt2current[1]) < 64 * 64)):
                    current[0] = (current[0] + pos[0]) / 2
                    current[1] = (current[1] + pos[1]) / 2

                    pos[0] = 0
                    pos[1] = -100

    return lines


def drawLine(line, img, rgb=(0, 0, 255)):
    if (line[1] != 0):

        m = -1 / np.tan(line[1])
        c = line[0] / np.sin(line[1])
        cv2.line(img, (0, int(c)), (int(img.shape[0]), int(m * img.shape[0] + c)), rgb)
    else:
        cv2.line(img, (line[0], 0), (line[0], img.shape[1]), rgb)


def nothing(x):
    pass


def calculateIntersections(outerBox, originalImage, topEdge, bottomEdge, leftEdge, rightEdge):
    height, width = outerBox.shape
    left1 = [0, 0];
    left2 = [0, 0]
    right1 = [0, 0];
    right2 = [0, 0]
    bottom1 = [0, 0];
    bottom2 = [0, 0]
    top1 = [0, 0];
    top2 = [0, 0]

    if leftEdge[1] != 0:
        left1[0] = 0;
        left1[1] = leftEdge[0] / sin(leftEdge[1])
        left2[0] = width;
        left2[1] = -left2[0] / tan(leftEdge[1]) + left1[0]
    else:
        left1[1] = 0;
        left1[0] = leftEdge[0] / cos(leftEdge[1])
        left2[1] = height;
        left2[0] = left1[0] - height * tan(leftEdge[1])

    if rightEdge[1] != 0:
        right1[0] = 0;
        right1[1] = rightEdge[0] / sin(rightEdge[1])
        right2[0] = width;
        right2[1] = -right2[0] / tan(rightEdge[1]) + right1[1]
    else:
        right1[1] = 0;
        right1[0] = rightEdge[0] / cos(rightEdge[1])
        right2[1] = height;
        right2[0] = right1[0] - height * tan(rightEdge[1])

    bottom1[0] = 0;
    bottom1[1] = bottomEdge[0] / sin(bottomEdge[1])
    bottom2[0] = width;
    bottom2[1] = -bottom2[0] / tan(bottomEdge[1]) + bottom1[1]

    top1[0] = 0;
    top1[1] = topEdge[0] / sin(topEdge[1])
    top2[0] = width;
    top2[1] = -top2[0] / tan(topEdge[1]) + top1[1]
    # znajdowanie przeciec linii

    leftA = left2[1] - left1[1]
    leftB = left1[0] - left2[0]

    leftC = leftA * left1[0] + leftB * left1[1]

    rightA = right2[1] - right1[1]
    rightB = right1[0] - right2[0]

    rightC = rightA * right1[0] + rightB * right1[1]

    topA = top2[1] - top1[1]
    topB = top1[0] - top2[0]

    topC = topA * top1[0] + topB * top1[1]

    bottomA = bottom2[1] - bottom1[1]
    bottomB = bottom1[0] - bottom2[0]

    bottomC = bottomA * bottom1[0] + bottomB * bottom1[1]

    detTopLeft = leftA * topB - leftB * topA;
    ptTopLeft = ((topB * leftC - leftB * topC) / detTopLeft, (leftA * topC - topA * leftC) / detTopLeft)

    detTopRight = rightA * topB - rightB * topA
    ptTopRight = ((topB * rightC - rightB * topC) / detTopRight, (rightA * topC - topA * rightC) / detTopRight)

    detBottomRight = rightA * bottomB - rightB * bottomA
    ptBottomRight = (
    (bottomB * rightC - rightB * bottomC) / detBottomRight, (rightA * bottomC - bottomA * rightC) / detBottomRight)

    detBottomLeft = leftA * bottomB - leftB * bottomA
    ptBottomLeft = (
    (bottomB * leftC - leftB * bottomC) / detBottomLeft, (leftA * bottomC - bottomA * leftC) / detBottomLeft)

    maxLength = (ptBottomLeft[0] - ptBottomRight[0]) * (ptBottomLeft[0] - ptBottomRight[0]) + (ptBottomLeft[1] -
                                                                                               ptBottomRight[1]) * (
                                                                                              ptBottomLeft[1] -
                                                                                              ptBottomRight[1])
    temp = (ptTopRight[0] - ptBottomRight[0]) * (ptTopRight[0] - ptBottomRight[0]) + (ptTopRight[1] - ptBottomRight[
        1]) * (ptTopRight[1] - ptBottomRight[1])

    if temp > maxLength:
        maxLength = temp
    temp = (ptTopRight[0] - ptTopLeft[0]) * (ptTopRight[0] - ptTopLeft[0]) + (ptTopRight[1] - ptTopLeft[1]) * (
    ptTopRight[1] - ptTopLeft[1]);

    if temp > maxLength:
        maxLength = temp

    temp = (ptBottomLeft[0] - ptTopLeft[0]) * (ptBottomLeft[0] - ptTopLeft[0]) + (ptBottomLeft[1] - ptTopLeft[1]) * (
    ptBottomLeft[1] - ptTopLeft[1])

    if temp > maxLength:
        maxLength = temp

    maxLength = int(sqrt(maxLength))

    src = np.array([
        ptTopLeft,
        ptTopRight,
        ptBottomRight,
        ptBottomLeft], dtype='float32')



    dst = np.array([
        [0, 0],
        [maxLength - 1, 0],
        [maxLength - 1, maxLength - 1],
        [0, maxLength - 1]], dtype='float32')

    undistorted = np.zeros((maxLength, maxLength), 'int')

    M = cv2.getPerspectiveTransform(src, dst)
    img = cv2.warpPerspective(originalImage, M, (maxLength, maxLength))

    # img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ret = cv2.adaptiveThreshold(
    #     img_grey,
    #     255,
    #     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #     cv2.THRESH_BINARY_INV,
    #     101, 1)
    return img, src


imgs2 = ['images/IMG01.jpg',
         'images/IMG02.jpg',
         'images/IMG04.jpg',]
images = ["sudoku-original.jpg"]
# Load an image

imgs2 = [cv2.imread(im) for im in imgs2]
images = [cv2.imread(img) for img in images]
# Resize The image
imgs2 = [imutils.resize(img, width=600,height=600) for img in imgs2]
images = [imutils.resize(img, width=600,height=600) for img in images]

def calculateEdges(lines):
    topEdge = (1000, 1000);
    topYIntercept = 100000;
    topXIntercept = 0
    botEdge = (-1000, -1000);
    botYIntercept = 0;
    botXIntercept = 0
    rigEdge = (1000, 1000);
    rigXIntercept = 0;
    rigYIntercept = 0
    lefEdge = (-1000, -1000);
    lefXIntercept = 100000;
    lefYIntercept = 0

    for i in range(len(lines)):
        current = lines[i][0]
        p = current[0]
        theta = current[1]
        if p == 0 and theta == -100:
            continue

        xIntercept = p / cos(theta)
        yIntercept = p / (cos(theta) * sin(theta))

        if theta > pi * 80 / 180 and theta < pi * 100 / 180:
            if p < topEdge[0]:
                topEdge = current
            if p > botEdge[0]:
                botEdge = current
        elif theta < pi * 10 / 180 or theta > pi * 170 / 180:
            if xIntercept > rigXIntercept:
                rigEdge = current
                rigXIntercept = xIntercept
            elif xIntercept <= lefXIntercept:
                lefEdge = current
                lefXIntercept = xIntercept
    return topEdge, botEdge, lefEdge, rigEdge

def processImages(images):
    # while (tempXASD < 1):
    #     tempXASD = 1
    # while (1):
    stack = []
    all = []
    processed = []
    i = 0
    for img in images:

        contour_list = []
        max = 1000

        clone = 0
        pic = 0

        clone = img.copy()

        # gray=clone

        # cv2.addWeighted(clone,i/10,np.zeros(img.shape,img.dtype),0,100)
        gray = cv2.cvtColor(cv2.addWeighted(clone, 2, np.zeros(img.shape, img.dtype), 0, 10), cv2.COLOR_BGR2GRAY)

        # cv2.imshow('norm', gray)
        # cv2.waitKey()

        gray = cv2.GaussianBlur(gray, (11, 11), 0)
        gray_threshed2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5, 2)
        kernel = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
        kernel = np.array(kernel)


        gray_threshed2 = cv2.dilate(gray_threshed2, kernel, iterations=1)
        _, contours, _ = cv2.findContours(gray_threshed2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)



        count = 0
        max_blob = -1
        max_blob_size = -1
        max_ptr = (0, 0)
        for y in range(0, len(gray_threshed2)):
            row = gray_threshed2[y]
            for x in range(0, len(row)):
                if (row[x] >= 128):
                    h, w = gray_threshed2.shape[:2]
                    mask = np.zeros((h + 2, w + 2), np.uint8)
                    area = cv2.floodFill(gray_threshed2, mask, (x, y), 64)

                    if (area[0] > max_blob):
                        max_ptr = (x, y)
                        max_blob = area[0]
        h, w = gray_threshed2.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(gray_threshed2, mask, max_ptr, 256)
        for y in range(0, len(gray_threshed2)):
            row = gray_threshed2[y]
            for x in range(0, len(row)):
                if (row[x] == 64 and x != max_ptr[0] and y != max_ptr[1]):
                    h, w = gray_threshed2.shape[:2]
                    mask = np.zeros((h + 2, w + 2), np.uint8)
                    area = cv2.floodFill(gray_threshed2, mask, (x, y), 0)
        gray_threshed2 = cv2.erode(gray_threshed2, kernel, iterations=1)

        cv2.imshow('norm',gray_threshed2)
        cv2.waitKey()

        # get out contour
        _, contours, hierarchy = cv2.findContours(gray_threshed2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # znajdue poprawnie ogolny kontur
        # teraz obraca

        rect = cv2.minAreaRect(contours[0])
        center = rect[0]
        angle = int (rect[2])
        # trza obrocic o ten kat !
        # gray_threshed2 = cv2.rotate(gray_threshed2,int (angle))
        # rows, cols = gray_threshed2.shape
        M = cv2.getRotationMatrix2D(center, angle, 1)

        gray_threshed2 = cv2.warpAffine(gray_threshed2, M, (600,600))
        imgCorColor = cv2.warpAffine(img, M, (600,600))


        print('rect', rect)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(img, [box], 0, (255, 0, 255), 2)

        # cv2.drawContours(img,contours,0,160)

        # cv2.imshow('norm2', gray_threshed2)
        cv2.imshow('norm', imgCorColor)
        cv2.waitKey()
        # rect = cv2.minAreaRect(cnt)


        lines = cv2.HoughLines(gray_threshed2, 1, np.pi / 180, 200)

        lines = mergeRelatedLines(lines, imgCorColor)


        # cv2.imshow('norm', img)
        # cv2.waitKey()

        topEdge, bottomEdge, leftEdge, rightEdge = calculateEdges(lines)

        drawLine(topEdge, gray_threshed2, (0, 0, 0))
        drawLine(bottomEdge, gray_threshed2, (0, 0, 0))
        drawLine(leftEdge, gray_threshed2, (0, 0, 0))
        drawLine(rightEdge, gray_threshed2, (0, 0, 0))

        # Liczenie skrtzyzowan linii

        # cv2.imshow('norm', gray_threshed2)
        # cv2.waitKey()


        print('before calcIntersections')

        cv2.imshow('norm', imgCorColor)
        cv2.waitKey()

        imgCorColor, corners = calculateIntersections(gray_threshed2, imgCorColor, topEdge, bottomEdge, leftEdge, rightEdge)

        cv2.imshow('norm', imgCorColor)
        cv2.waitKey()
        print('after calcIntersections')

        # detect corners
        # tempImg = gray_threshed2
        # for ite in range(0,100):
        #
        #     corners = cv2.goodFeaturesToTrack(gray_threshed2, 3, 0.01, 10)
        #     corners = np.int0(corners)
        #
        #     corns = []
        #     # print(corners)
        #     for corner in corners:
        #         x, y = corner.ravel()
        #         corns.append([x,y])
        #     print(corns)
        #         # cv2.circle(gray_threshed2, (x, y), 10, 255, -1)
        #     # cv2.fillPoly(gray_threshed2,corners,255)

        # print(corners)
        # corners = [[int (x), int (y)] for [x,y] in corners]
        # print(corners)
        # cv2.fillConvexPoly(gray_threshed2,corners,255)
        #

        # cv2.imshow('norm2', imgCorColor)





        img_grey = cv2.cvtColor(imgCorColor, cv2.COLOR_BGR2GRAY)

        imgCor = cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 101, 1)



        h, w = imgCor.shape
        # print(h,w)
        h = int(h / 9)
        w = int(w / 9)
        # print(h,w)

        imgsArr = []
        for i in range(0, 9):
            imgsArr.append([])
            # imgsArr[i] = []
            for j in range(0, 9):
                tempImg = imgCor[i * h: i * h + h, j * w: j * w + w]
                tempImg = cv2.erode(tempImg, kernel)
                imgsArr[i].append(tempImg)
                # imgCor[i*h : i*h+h, j*w: j*w+w] = tempImg

        # imgCor = cv2.erode(imgCor,kernel)

        # imgCor = cv2.morphologyEx(imgCor, cv2.MORPH_OPEN, kernel)

        samples = np.empty((0, 100))
        responses = []
        keys = [i for i in range(48, 58)]

        # gray = cv2.cvtColor(imgCor, cv2.COLOR_BGR2GRAY)
        # blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

        # numberContours, hierarchy  = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        imgCor, numberContours, hierarchy = cv2.findContours(imgCor, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        print(len(numberContours))

        imgH, imgW = imgCor.shape

        # print('h,w',h,w)
        imgH = int(imgH / 9)
        imgW = int(imgW / 9)
        boxAreaMax = imgH*imgW*0.6
        boxAreaMin = imgH*imgW*0.05
        print('Box min:',boxAreaMin,', max:',boxAreaMax)



        for cnt in numberContours:

            # if cv2.contourArea(cnt) > 50 and cv2.contourArea(cnt) < 1500:
            if cv2.contourArea(cnt) > boxAreaMin and cv2.contourArea(cnt)<boxAreaMax:
                print(cv2.contourArea(cnt))
                [x, y, w, h] = cv2.boundingRect(cnt)


                if not (h*3<w) and not(w*3 < h):
                    numberImgBW = imgCorColor[y-2:y+h+2,x-2:x+w+2]
                    numberImgBW = cv2.resize(numberImgBW, (imgW, imgH))
                    numberImgBW = cv2.cvtColor(numberImgBW,cv2.COLOR_BGR2GRAY)
                    # numberImgBW = cv2.cvtColor(numberImgBW,cv2.BIN)


                    cv2.rectangle(imgCorColor, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    roi = imgCor[x:x + w,y:y + h]
                    roismall = cv2.resize(roi, (10, 10))

                    M = cv2.moments(cnt)
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])

                    xBox = int(cX / imgH)
                    yBox = int(cY / imgW)

                    # get corresponding img form array
                    cv2.imshow('temp',numberImgBW)

                    print('x:', cX, ', y:', cY, 'box(', int(cX / imgH) + 1, ',', int(cY / imgW) + 1, ') w:',w,'h:',h, 'w*h',w*h)

                    cv2.circle(imgCorColor, (cX, cY), 2, (255, 255, 255), -1)

                    cv2.imshow('norm', imgCorColor)
                    cv2.waitKey()



                    # print(cv2.contourArea(cnt))
                    # key = cv2.waitKey(0)
                    # if key == 27:  # (escape to quit)
                    #     sys.exit()
                    # elif key in keys:
                    #     responses.append(int(chr(key)))
                    #     sample = roismall.reshape((1, 100))
                    #     samples = np.append(samples, sample, 0)

        # responses = np.array(responses,np.float32)
        # responses = responses.reshape((responses.size,1))

        # print
        # for i in range(0, len(lines)):
        #     drawLine(lines[i][0], gray_threshed2, (128, 0, 128))
        pic = gray_threshed2

        # Draw contours on the original image

        # there is an outer boundary and inner boundary for each eadge, so contours double

        # processed.append(pic)
        processed.append(imgCor)

    stack_1 = np.vstack((processed))

    cv2.imshow('Objects Detected_1', processed[0])

    cv2.waitKey()

    cv2.destroyAllWindows()

# processImages(images)
processImages(imgs2)