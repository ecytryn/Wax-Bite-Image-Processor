import cv2
import os
import pandas as pd
import numpy as np
from pynput.keyboard import Key, Controller
from utils import Tooth, Match, CONFIG


GREY = (100,100,100)
REC_THICKNESS = 2
STORE_MODE = Tooth.TOOTH
MODE = Tooth.TOOTH
MODES = [mode for mode in Tooth if mode != Tooth.NO_BOX]


x = np.array([])
y = np.array([])
w = np.array([])
h = np.array([])
type = np.array([])
clone = 0


def GUI(fileName, imgName, fileType, mode):
    global x, y, w, h, type, clone, MODE, STORE_MODE

    MODE = Tooth.TOOTH
    STORE_MODE = Tooth.TOOTH

    if mode == Match.ONE_D:
        imgPath = os.path.join("processed", "projection", imgName, f"projection{fileType}")
        manualDataPath = os.path.join("processed", "manual", imgName, "manual data 1D.csv")
        if os.path.isfile(manualDataPath):
            imgDataPath = manualDataPath
        else:
            imgDataPath = os.path.join("processed", "template matching", imgName, "template matching 1D.csv")
    else:
        imgPath = os.path.join("img", fileName)
        manualDataPath = os.path.join("processed", "manual", imgName, f"manual data.csv")
        if os.path.isfile(manualDataPath):
            imgDataPath = manualDataPath
        else:
            imgDataPath = os.path.join("processed", "template matching", imgName, f"template matching.csv") 

    if not os.path.isfile(imgPath):
        raise RuntimeError(f"{imgPath} does not exist. A hyperbola fit was likely not found or did you run fitProject first?")
    if not os.path.isfile(imgDataPath):
        print(f"{imgDataPath} not found, empty data set used")
    else:
        df = pd.read_csv(imgDataPath)
        x = df["x"].to_numpy()
        y = df["y"].to_numpy()
        w = df["w"].to_numpy()
        h = df["h"].to_numpy()
        if "type" not in df.columns:
            df["type"] = [Tooth.TOOTH for _ in range(len(df.index))]
            type = df["type"]
        else: 
            type = np.array([])
            for i in df["type"]:
                if i == "Tooth.TOOTH":
                    type = np.append(type, Tooth.TOOTH)
                elif i == "Tooth.GAP":
                    type = np.append(type, Tooth.GAP)
                elif i == "Tooth.CENTER_T":
                    type = np.append(type, Tooth.CENTER_T)
                elif i == "Tooth.CENTER_G":
                    type = np.append(type, Tooth.CENTER_G)
    image = cv2.imread(imgPath)
    cv2.namedWindow(imgName)
    cv2.setMouseCallback(imgName, leftClick)
    modeIndex = 0

    while 1:
        clone = image.copy()
        for i in range(len(x)):
            clone = drawTooth(clone, x[i], y[i], w[i], h[i], type[i])

        # borderImg = cv2.copyMakeBorder(clone, 1000, 1000, 0, 0, cv2.BORDER_CONSTANT, value=GREY)
        if MODE == Tooth.TOOTH:
            textImg = cv2.putText(clone, "mode: tooth", (10,40), cv2.FONT_HERSHEY_DUPLEX, 1.5, CONFIG.TOOTH, 2)
        elif MODE == Tooth.GAP: 
            textImg = cv2.putText(clone, "mode: gap", (10,40), cv2.FONT_HERSHEY_DUPLEX, 1.5, CONFIG.GAP, 2)
        elif MODE == Tooth.CENTER_T: 
            textImg = cv2.putText(clone, "mode: center tooth", (10,40), cv2.FONT_HERSHEY_DUPLEX, 1.5, CONFIG.CENTER, 2)
        elif MODE == Tooth.CENTER_G:
            textImg = cv2.putText(clone, "mode: center gap", (10,40), cv2.FONT_HERSHEY_DUPLEX, 1.5, CONFIG.CENTER, 2)
        else:
            textImg = cv2.putText(clone, "mode: viewing", (10,40), cv2.FONT_HERSHEY_DUPLEX, 1.5, GREY, 2)

        cv2.imshow(imgName, textImg)
        key = cv2.waitKey(0)
        if key == 32:
            if MODE == Tooth.NO_BOX:
                MODE = STORE_MODE
            else:
                STORE_MODE = MODE
                MODE = Tooth.NO_BOX
        elif key == 27:
            cv2.destroyAllWindows()
            break
        elif key == 9 and MODE != Tooth.NO_BOX:
            modeIndex = (modeIndex + 1) % len(MODES)
            MODE = MODES[modeIndex]
        elif key == ord("s"):
            save(fileName, imgName, fileType, mode, clone)
            break
        elif key == ord("1"):
            MODE = Tooth.TOOTH
            modeIndex = 0
        elif key == ord("2"):
            MODE = Tooth.GAP
            modeIndex = 1
        elif key == ord("3"):
            MODE = Tooth.CENTER_T
            modeIndex = 2
        elif key == ord("4"):
            MODE = Tooth.CENTER_G
            modeIndex = 3

def drawTooth(image, x, y, w, h, mode):
    OFFSET_X = -5
    OFFSET_Y = 5

    centerX = int(x + 1/2 * w)
    centerY = int(y + 1/2 * h)
    endX = x + w
    endY = y + h

    if mode == Tooth.TOOTH:
        color = CONFIG.TOOTH
    elif mode == Tooth.GAP:
        color = CONFIG.GAP
    elif mode == Tooth.CENTER_T:
        color = CONFIG.CENTER
        imageLabelled = drawLabel(image, centerX+OFFSET_X, centerY+OFFSET_Y, color, "T")
    elif mode == Tooth.CENTER_G:
        color = CONFIG.CENTER
        imageLabelled = drawLabel(image, centerX+OFFSET_X, centerY+OFFSET_Y, color, "G")
    imageLabelled = drawCenter(image, centerX, centerY, color)

    if MODE == Tooth.NO_BOX:
        return imageLabelled

    imageRec = drawRectangle(imageLabelled, x, y, endX, endY, color)
    return imageRec

def drawRectangle(image, x, y, endX, endY, COLOR):
    THICKNESS = 2
    newImage = cv2.rectangle(image, pt1=(x,y), pt2=(endX,endY), color=COLOR, thickness=REC_THICKNESS)
    return newImage

def drawCenter(image, centerX, centerY, COLOR):
    newImage = cv2.circle(image, (centerX, centerY), radius=5, color=COLOR, thickness=-1)
    return newImage

def drawLabel(image, x, y, color, label):
    newImage = cv2.putText(image, label, (x,y), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)
    return newImage

def leftClick(event, clickedX, clickedY, flags, params):

    global x, y, w, h, type
    if event == cv2.EVENT_LBUTTONUP and MODE != Tooth.NO_BOX:
        draw = True
        for index in range(len(x)):
            if clickedX >= x[index] and clickedX <= x[index]+w[index] and clickedY >= y[index] and clickedY <= y[index]+h[index]:
                x = np.delete(x, index)
                y = np.delete(y, index)
                w = np.delete(w, index)
                h = np.delete(h, index)
                type = np.delete(type, index)
                draw = False
                break

        if draw and (MODE == Tooth.CENTER_T or MODE == Tooth.CENTER_G):
            centerT = np.where(type == Tooth.CENTER_T)
            for index in centerT[0]:
                x = np.delete(x, index)
                y = np.delete(y, index)
                w = np.delete(w, index)
                h = np.delete(h, index)
                type = np.delete(type, index)

            centerG = np.where(type == Tooth.CENTER_G)
            for index in centerG[0]:
                x = np.delete(x, index)
                y = np.delete(y, index)
                w = np.delete(w, index)
                h = np.delete(h, index)
                type = np.delete(type, index)

        if draw:
            newX = int(clickedX - 1/2 * CONFIG.SQUARE)
            newY = int(clickedY - 1/2 * CONFIG.SQUARE)
            x = np.append(x, newX)
            y = np.append(y, newY)
            w = np.append(w, CONFIG.SQUARE)
            h = np.append(h, CONFIG.SQUARE)
            type = np.append(type, MODE)

        keyboard = Controller()
        keyboard.press("a")
        keyboard.release("a")

def save(fileName, imgName, fileType, mode, image, dfRes = None):
    if mode == Match.ONE_D:  
        PATH_IMG = os.path.join("processed", "manual", imgName, f"manual 1D{fileType}")
        PATH_DATA = os.path.join("processed", "manual", imgName, f"manual data 1D.csv")
    else:
        PATH_IMG = os.path.join("processed", "manual", imgName, f"manual{fileType}")
        PATH_DATA = os.path.join("processed", "manual", imgName, f"manual data.csv")
    cv2.imwrite(PATH_IMG, image)

    if isinstance(dfRes, pd.DataFrame):
        dfRes.to_csv(PATH_DATA)
    else:
        df = pd.DataFrame()
        df["x"] = x
        df["y"] = y
        df["w"] = w
        df["h"] = h
        df["type"] = type
        df.sort_values(by=["x"], inplace=True)
        df.to_csv(PATH_DATA)
    cv2.destroyAllWindows()



def plotPreviousData(fileName, imgName, fileType, mode, df):

    if mode == Match.ONE_D:
        imgPath = os.path.join("processed", "projection", imgName, f"projection{fileType}")
    else:
        imgPath = os.path.join("img", fileName)

    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    w = df["w"].to_numpy()
    h = df["h"].to_numpy()
    type = np.array([])
    for i in df["type"]:
        if i == "Tooth.TOOTH":
            type = np.append(type, Tooth.TOOTH)
        elif i == "Tooth.GAP":
            type = np.append(type, Tooth.GAP)
        elif i == "Tooth.CENTER_T":
            type = np.append(type, Tooth.CENTER_T)
        elif i == "Tooth.CENTER_G":
            type = np.append(type, Tooth.CENTER_G)

    if not os.path.isfile(imgPath):
        raise RuntimeError(f"{imgPath} does not exist. A hyperbola fit was likely not found or did you run fitProject first?")

    image = cv2.imread(imgPath)
    for i in range(len(x)):
        image = drawTooth(image, int(x[i]-1/2*w[i]), int(y[i]-1/2*h[i]), w[i], h[i], type[i])

    save(fileName, imgName, fileType, mode, image, df)