import cv2
import numpy as np
from numpy import tan, cos, sin, pi, fabs, sqrt, math
import imutils
from sklearn.externals import joblib
from skimage.feature import hog
from scipy.spatial.distance import cdist
import digit_regoc as recognizer

showSteps = False

clf = joblib.load("digits_cls.pkl")

model = recognizer.setUp()



# tablica z intami


def isValidLine(line):
    # czy z zakresu 0-9
    for i in range(len(line)):
        if line[i] < 0 or line[i] > 9:
            return False

    # czy sa duplikaty od 1-9
    tempVal = []
    for i in range(0, 9):
        tempVal.append(0)
    for i in line:
        if i == 0:
            continue
        tempVal[i - 1] = tempVal[i - 1] + 1
        if tempVal[i - 1] > 1:
            return False
    return True


def areValidBoxes(board):
    box = []
    for i in range(0, 3):
        for j in range(0, 3):
            box.clear()
            for k in range(0, 9):
                box.append(board[int(i * 3 + k / 3)][int(j * 3 + k % 3)])
            if not (isValidLine(box)): return False
    return True


def isBoardValid(board):
    row = []
    col = []
    for i in range(0, 9):
        row.clear()
        col.clear()
        for j in range(0, 9):
            row.append(board[i][j])
            col.append(board[j][i])
        if not (isValidLine(row)) or not (isValidLine(col)): return False
    return areValidBoxes(board)


def solve(x, y, board):
    if (board[x][y] != 0):
        if isBoardValid(board):
            if x == 8 and y == 8:
                return True, board
            nextX = x + 1
            nextY = y
            if nextX >= 9:
                nextX = 0
                nextY = nextY + 1
            wynik, board = solve(nextX, nextY, board)
            return wynik, board
        else:
            return False, board
    for val in range(1, 10):
        board[x][y] = val
        if isBoardValid(board):
            if x == 8 and y == 8:
                return True, board
            nextX = x + 1
            nextY = y
            if nextX >= 9:
                nextX = 0
                nextY = nextY + 1
            wynik, board = solve(nextX, nextY, board)
            if wynik:
                return True, board
    board[x][y] = 0
    return False, board


def solveBoard(board):
    wynik, board = solve(0, 0, board)
    return wynik, board


def mergeRelatedLines(lines, img):
    # print(img.size)
    height, width = img.shape
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


def calculateIntersections(outerBox, originalImage, src):
    ptTopLeft = src[0]
    ptTopRight = src[1]
    ptBottomRight = src[2]
    ptBottomLeft = src[3]
    # print('top:', topEdge, ', bot:', bottomEdge, ', left:', leftEdge, ', right:', rightEdge)
    height, width = outerBox.shape
    # left1 = [0, 0];
    # left2 = [0, 0]
    # right1 = [0, 0];
    # right2 = [0, 0]
    # bottom1 = [0, 0];
    # bottom2 = [0, 0]
    # top1 = [0, 0];
    # top2 = [0, 0]
    #
    # if leftEdge[1] != 0:
    #     left1 = [0, leftEdge[0] / sin(leftEdge[1])]
    #     left2 = [width, -width / tan(leftEdge[1]) + left1[1]]
    # else:
    #     left1 = [leftEdge[0] / cos(leftEdge[1]), 0]
    #     left2 = [height - height * tan(leftEdge[1]), height]
    #
    # if rightEdge[1] != 0:
    #     right1 = [0, rightEdge[0] / sin(rightEdge[1])]
    #     right2 = [width, -width / tan(rightEdge[1]) + right1[1]]
    # else:
    #     right1 = [rightEdge[0] / cos(rightEdge[1]), 0]
    #     right2 = [right1[0] - height * tan(rightEdge[1]), height]
    #
    # bottom1 = [0, bottomEdge[0] / sin(bottomEdge[1])]
    # bottom2 = [width, -width / tan(bottomEdge[1]) + bottom1[1]]
    #
    # top1 = [0, topEdge[0] / sin(topEdge[1])]
    # top2 = [width, -width / tan(topEdge[1]) + top1[1]]
    #
    # # znajdowanie przeciec linii
    #
    # leftA = left2[1] - left1[1]
    # leftB = left1[0] - left2[0]
    #
    # leftC = leftA * left1[0] + leftB * left1[1]
    #
    # rightA = right2[1] - right1[1]
    # rightB = right1[0] - right2[0]
    #
    # rightC = rightA * right1[0] + rightB * right1[1]
    #
    # topA = top2[1] - top1[1]
    # topB = top1[0] - top2[0]
    #
    # topC = topA * top1[0] + topB * top1[1]
    #
    # bottomA = bottom2[1] - bottom1[1]
    # bottomB = bottom1[0] - bottom2[0]
    #
    # bottomC = bottomA * bottom1[0] + bottomB * bottom1[1]
    #
    # detTopLeft = leftA * topB - leftB * topA;
    # ptTopLeft = ((topB * leftC - leftB * topC) / detTopLeft, (leftA * topC - topA * leftC) / detTopLeft)
    #
    # detTopRight = rightA * topB - rightB * topA
    # ptTopRight = ((topB * rightC - rightB * topC) / detTopRight, (rightA * topC - topA * rightC) / detTopRight)
    #
    # detBottomRight = rightA * bottomB - rightB * bottomA
    # ptBottomRight = (
    #     (bottomB * rightC - rightB * bottomC) / detBottomRight, (rightA * bottomC - bottomA * rightC) / detBottomRight)
    #
    # detBottomLeft = leftA * bottomB - leftB * bottomA
    # ptBottomLeft = (
    #     (bottomB * leftC - leftB * bottomC) / detBottomLeft, (leftA * bottomC - bottomA * leftC) / detBottomLeft)

    #

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

    tempImg = originalImage

    # for point in src:
    cv2.circle(tempImg, (ptTopLeft[0], ptTopLeft[1]), 2, (0, 255, 0))
    cv2.circle(tempImg, (ptTopRight[0], ptTopRight[1]), 2, (255, 255, 0))
    cv2.circle(tempImg, (ptBottomRight[0], ptBottomRight[1]), 2, (0, 0, 255))
    cv2.circle(tempImg, (ptBottomLeft[0], ptBottomLeft[1]), 2, (0, 255, 255))

    # cv2.imshow('norm', tempImg)
    # cv2.waitKey()

    dst = np.array([
        [0, 0],
        [maxLength - 1, 0],
        [maxLength - 1, maxLength - 1],
        [0, maxLength - 1]], dtype='float32')

    undistorted = np.zeros((maxLength, maxLength), 'int')

    M = cv2.getPerspectiveTransform(src, dst)
    img = cv2.warpPerspective(originalImage, M, (maxLength + 1, maxLength + 1))

    # img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ret = cv2.adaptiveThreshold(
    #     img_grey,
    #     255,
    #     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #     cv2.THRESH_BINARY_INV,
    #     101, 1)
    return img, src


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


def calculateBoardCorners(bwImg):
    _, contours, _ = cv2.findContours(bwImg.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    h, w = bwImg.shape
    # print(contours[0])
    srcPoints = [
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]]
    dstPoints = [closest_node(point, contours[0]) for point in srcPoints]

    # print(dstPoints)

    return dstPoints, srcPoints



def drawNumbersOnImage(img, numbers):
    w, h, _ = img.shape
    w = w / 9
    h = h / 9
    finalNumbers = []
    for i in range(9):
        finalNumbers.append([])
        for j in range(9):
            finalNumbers[i].append(0)

    for i in range(9):
        for j in range(9):
            if numbers[i][j] != 0:
                # finalNumbers[i][j] = numbers[i][j][1]  # biore kNN
                finalNumbers[i][j]  = numbers[i][j][0]  # biore SVM

    wynik, board = solveBoard(finalNumbers)
    if wynik:
        print('Znaleziono rozwiazanie !')
    else:
        print('Nie znaleziono rozwiazanie !')
    # SVM, kNN, x, y+h
    for i in range(0, 9):
        for j in range(0, 9):
            # BGR
            color = (0, 0, 255)
            grubosc = 2
            skala = 0.8
            x = int(i * w + (w / 2))
            y = int(j * h + (h / 2))
            if wynik:
                cv2.putText(img, str(board[i][j]), (x,y), cv2.FONT_HERSHEY_SIMPLEX, skala, color,
                            grubosc)
                showStepsImgs(img)
            elif finalNumbers[i][j] != 0:
                x = int(i*w + (w/2))
                y = int (j*h + (h/2))
                cv2.putText(img, str(finalNumbers[i][j]) + '', (x, y), cv2.FONT_HERSHEY_SIMPLEX, skala, color, grubosc)  # kolor w skali BGR

    return img
def predictNumber(numberImg):
    roi_hog_fd = hog(numberImg, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
    nbr = clf.predict(np.array([roi_hog_fd], 'float64'))
    return int(nbr[0])


def closest_node(node, nodes):
    tempNodes = []
    for i in range(len(nodes)):
        tempNodes.append([nodes[i][0][0], nodes[i][0][1]])
    x = nodes[cdist([node], tempNodes).argmin()]
    return [x[0][0], x[0][1]]


def rotateImgs(imgBW, img, angleRotation=0):
    if angleRotation != 0:
        w, h = imgBW.shape

        M = cv2.getRotationMatrix2D((w / 2, h / 2), angleRotation, 1)
        imgCorColor = cv2.warpAffine(img, M, (600, 600))
        return imgBW, imgCorColor, 0
    else:
        _, contours, hierarchy = cv2.findContours(imgBW, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        rect = cv2.minAreaRect(contours[0])
        center = rect[0]
        angle = int(rect[2])
        angleCpy = angle
        # print('Kat nachylenia ', angle)
        if abs(angleCpy) > 45:
            angleCpy = 90 - abs(angleCpy)

        M = cv2.getRotationMatrix2D(center, angleCpy, 1)

        imgBW = cv2.warpAffine(imgBW, M, (600, 600))
        imgCorColor = cv2.warpAffine(img, M, (600, 600))
        return imgBW, imgCorColor, angle


def predictDigits(imgBW, imgColor):
    numbers = []
    for i in range(0, 9):
        numbers.append([])
        for j in range(0, 9):
            numbers[i].append(0)

    imgBW, numberContours, hierarchy = cv2.findContours(imgBW, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    imgH, imgW = imgBW.shape
    imgH = int(imgH / 9)
    imgW = int(imgW / 9)
    boxAreaMax = imgH * imgW * 0.7
    boxAreaMin = imgH * imgW * 0.04
    tempImg = 0
    lezace = stojace = 0
    # cv2.imshow('img',imgColor)
    for cnt in numberContours:
        if cv2.contourArea(cnt) > boxAreaMin and cv2.contourArea(cnt) < boxAreaMax:
            [x, y, w, h] = cv2.boundingRect(cnt)

            # if not (h * 3 < w) and not (w * 3 < h):
            if not (h * 4 < w):
                if h < w:
                    lezace = lezace + 1
                else:
                    stojace = stojace + 1

                # obrobka malej liczby
                numberImgBW = tempImg
                numberImgBW = imgColor[y - 1:y + h + 1, x - 1:x + w + 1]

                numberImgBW = cv2.cvtColor(numberImgBW, cv2.COLOR_BGR2GRAY)
                # numberImgBW = cv2.GaussianBlur(numberImgBW, (11, 11), 0)
                numberImgBW = cv2.adaptiveThreshold(numberImgBW, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                        cv2.THRESH_BINARY_INV, 5, 2)
                kernel = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
                kernel = np.array(kernel)

                # numberImgBW = cv2.dilate(numberImgBW, np.ones((3, 3), np.uint8), iterations=1)
                numberImgBW = cv2.erode(numberImgBW, kernel, iterations=1)
                numberImgBW = cv2.dilate(numberImgBW, kernel, iterations=1)
                numberImgBW = cv2.resize(numberImgBW, (28, 28), interpolation=cv2.INTER_AREA)
                for i in range(0, 28):
                    numberImgBW[i, 0] = 0
                    numberImgBW[i, 1] = 0
                    numberImgBW[0, i] = 0
                    numberImgBW[1, i] = 0

                # do wykrycia w ktorej komorce leze dana liczba
                M = cv2.moments(cnt)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                xBox = int(cX / imgH)
                yBox = int(cY / imgW)

                # kNN

                number1 = recognizer.predictNumber(numberImgBW, model)
                # SVM
                number2 = predictNumber(numberImgBW)

                numbers[xBox][yBox] = [number2, number1, x, y + h]
                # print('kNN:', number1, ',SVM:', number2)
                # showAndWait(numberImgBW,param='num', wait=True)

    return imgColor, (stojace - lezace), numbers



def preprocessImage(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray_threshed2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5, 2)
    kernel = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
    kernel = np.array(kernel)

    gray_threshed2 = cv2.dilate(gray_threshed2, kernel, iterations=1)
    _, contours, _ = cv2.findContours(gray_threshed2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print('.')
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
    return gray_threshed2


def showAndWait(img, param='_', wait=False):
    cv2.imshow(param, img)
    if wait == True:
        cv2.waitKey()


def showStepsImgs(img, img2=None):
    if showSteps == True:
        if img2 is not None:
            showAndWait(img2, '1')
            showAndWait(img, wait=True)
        else:
            showAndWait(img, wait=True)
        cv2.destroyWindow('1')


def processImages(images,imagesNames):
    for imgNumer,img in enumerate(images):

        showStepsImgs(img)
        imgBW = preprocessImage(img.copy())
        showStepsImgs(imgBW)

        # Zalozenie: nie moze byc do gory nogami zdjecie ! maxymalny kat obrocenia planszy 90 stopni
        # jeśli kontury wyjda lezace a nie stojace powtorzyc wszystko pod tym i zmienic kat na +-90
        imgBW, imgCorColor, angle = rotateImgs(imgBW, img)
        showStepsImgs(imgCorColor, imgBW)

        src, dst = calculateBoardCorners(imgBW)

        # print('asd')
        imgCorColor, corners = calculateIntersections(imgBW, imgCorColor, src)
        showStepsImgs(imgCorColor, imgBW)

        imgBW = cv2.cvtColor(imgCorColor, cv2.COLOR_BGR2GRAY)

        imgBW = cv2.adaptiveThreshold(imgBW, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 101, 2)

        imgCorColor, czyObrocic, numbers = predictDigits(imgBW, imgCorColor)

        if czyObrocic < 0:
            w, h, _ = imgCorColor.shape
            pts1 = np.float32([[0, 0], [0, w], [h, w], [h, 0]])
            if angle < 0:
                pts2 = np.float32([[0, w], [h, w], [h, 0], [0, 0]])
                print('Trzeba obrocic w lewo (90 stopni)')
            else:
                pts2 = np.float32([[h, 0], [0, 0], [0, w], [h, w]])
                print('Trzeba obrocic w prawo (-90 stopni)')

            imgCorColor = cv2.warpPerspective(imgCorColor, cv2.getPerspectiveTransform(pts1, pts2), (w, h))
            imgBW = cv2.warpPerspective(imgBW, cv2.getPerspectiveTransform(pts1, pts2), (w, h))

            # numbers = 0
            imgCorColor, czyObrocic, numbers = predictDigits(imgBW, imgCorColor)

        showStepsImgs(imgCorColor, imgBW)
        imgCorColor = drawNumbersOnImage(imgCorColor, numbers)
        showStepsImgs(imgCorColor)

        # solve sudoku with numbers - trzeba  przerobic 'numbers'

        #

        if showSteps == False:
            print('Zdjecie: ',imageNames[imgNumer])
            # cv2.destroyAllWindows()
            showAndWait(imgCorColor,param='Edited', wait=False)
            showAndWait(img,param='Orginal' ,wait=True)

    # stack_1 = np.vstack((processed))

    # cv2.imshow('Objects Detected_1', processed[0])

    # cv2.waitKey()
    cv2.destroyAllWindows()



def setUpTestImgs():
    imageNames = ['images/sudoku_to_solve.jpg',
              'images/IMG01.jpg',
              'images/IMG02.jpg',
              'images/IMG03.jpg',
              'images/IMG04.jpg',
              'images/IMG05.jpg',
              'images/article.jpg',
              'images/sudoku-original.jpg',
              'images/sudoku.jpg',
              'images/dataset-card.jpg',
              'images/DiabolicalPuzzle.jpg',
              'images/sud.jpg',
              'images/sudoku1.jpg',
              'images/Sudoku-900x900.jpg',
              'images/sudoku4.jpg',
              'images/easy sudoku to print.jpg',
              'images/sudoku_SP56c.jpg',
              'images/Sudoku3324.jpg',
              'images/sudoku-ok.jpg',
              'images/sudoku-gratuit.jpg',
              'images/sudoku5.jpg',
              'images/6_1.jpg',
              'images/6_2.jpg',
              'images/6_3.jpg',
              'images/6_4.jpg',
              'images/6_5.jpg',
              'images/6_6.jpg'
              ]
    images = [cv2.imread(im) for im in imageNames]

    images = [imutils.resize(img, width=600) for img in images]
    return images, imageNames


if __name__ == '__main__':
    showSteps = False
    # showSteps = True

    images, imageNames = setUpTestImgs()
    processImages(images, imageNames)
    # processImages(images)
