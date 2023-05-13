import cv2
import os
import pandas as pd
import numpy as np
from pynput.keyboard import Key, Controller
from utils import Tooth, Match, CONFIG


GREY = (100,100,100)
REC_THICKNESS = 2
SQUARE = 30
STORE_MODE = Tooth.TOOTH
MODE = Tooth.TOOTH
MODES = [mode for mode in Tooth if mode != Tooth.NO_BOX]


x = np.array([])
y = np.array([])
w = np.array([])
h = np.array([])
type = np.array([])
clone = 0


def GUI(file_name, img_name, mode):
    global x, y, w, h, type, clone, MODE, STORE_MODE

    MODE = Tooth.TOOTH
    STORE_MODE = Tooth.TOOTH

    if mode == Match.ONE_D:
        IMG_PATH = os.path.join("processed", "projection", file_name)
        IMG_DATA_PATH = os.path.join("processed", "match data 1D", f"{img_name}.csv")
    else:
        IMG_PATH = os.path.join("img", file_name)
        IMG_DATA_PATH = os.path.join("processed", "match data", f"{img_name}.csv") 

    if not os.path.isfile(IMG_PATH):
        raise RuntimeError(f"{IMG_PATH} does not exist. A hyperbola fit was likely not found or did you run fit_project first?")
    if not os.path.isfile(IMG_DATA_PATH):
        print(f"{IMG_DATA_PATH} not found, empty data set used")
    else:
        df = pd.read_csv(IMG_DATA_PATH)
        df["type"] = [Tooth.TOOTH for _ in range(len(df.index))]
        x = df["x"].to_numpy()
        y = df["y"].to_numpy()
        w = df["w"].to_numpy()
        h = df["h"].to_numpy()
        type = df["type"].to_numpy()

    image = cv2.imread(IMG_PATH)
    cv2.namedWindow(img_name)
    cv2.setMouseCallback(img_name, left_click)
    mode_index = 0

    while 1:
        clone = image.copy()
        for i in range(len(x)):
            draw_tooth(clone, x[i], y[i], w[i], h[i], type[i])

        # border_img = cv2.copyMakeBorder(clone, 1000, 1000, 0, 0, cv2.BORDER_CONSTANT, value=GREY)
        if MODE == Tooth.TOOTH:
            text_img = cv2.putText(clone, "mode: tooth", (10,40), cv2.FONT_HERSHEY_DUPLEX, 1.5, CONFIG.TOOTH, 2)
        elif MODE == Tooth.GAP: 
            text_img = cv2.putText(clone, "mode: gap", (10,40), cv2.FONT_HERSHEY_DUPLEX, 1.5, CONFIG.GAP, 2)
        elif MODE == Tooth.CENTER_T: 
            text_img = cv2.putText(clone, "mode: center tooth", (10,40), cv2.FONT_HERSHEY_DUPLEX, 1.5, CONFIG.CENTER, 2)
        elif MODE == Tooth.CENTER_G:
            text_img = cv2.putText(clone, "mode: center gap", (10,40), cv2.FONT_HERSHEY_DUPLEX, 1.5, CONFIG.CENTER, 2)
        else:
            text_img = cv2.putText(clone, "mode: viewing", (10,40), cv2.FONT_HERSHEY_DUPLEX, 1.5, GREY, 2)

        cv2.imshow(img_name, text_img)
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
            mode_index = (mode_index + 1) % len(MODES)
            MODE = MODES[mode_index]
        elif key == ord("s"):
            save(file_name, img_name, mode)
            break
        elif key == ord("1"):
            MODE = Tooth.TOOTH
            mode_index = 0
        elif key == ord("2"):
            MODE = Tooth.GAP
            mode_index = 1
        elif key == ord("3"):
            MODE = Tooth.CENTER_T
            mode_index = 2
        elif key == ord("4"):
            MODE = Tooth.CENTER_G
            mode_index = 3

def draw_tooth(image, x, y, w, h, mode):
    OFFSET_X = -5
    OFFSET_Y = 5

    center_x = int(x + 1/2 * w)
    center_y = int(y + 1/2 * h)
    end_x = x + w
    end_y = y + h

    label = "0"

    if mode == Tooth.TOOTH:
        color = CONFIG.TOOTH
    elif mode == Tooth.GAP:
        color = CONFIG.GAP
    elif mode == Tooth.CENTER_T :
        color = CONFIG.CENTER
    elif mode == Tooth.CENTER_G:
        color = CONFIG.CENTER

    
    if mode == Tooth.CENTER_G:
        image_labelled = draw_label(image, center_x+OFFSET_X, center_y+OFFSET_Y, color, "G")
    elif mode == Tooth.CENTER_T:
        image_labelled = draw_label(image, center_x+OFFSET_X, center_y+OFFSET_Y, color, "T")
    else: 
        image_labelled = draw_center(image, center_x, center_y, color)

    if MODE == Tooth.NO_BOX:
        return image_labelled

    image_rec = draw_rectangle(image_labelled, x, y, end_x, end_y, color)
    return image_rec

def draw_rectangle(image, x, y, end_x, end_y, COLOR):
    THICKNESS = 2
    new_img = cv2.rectangle(image, pt1=(x,y), pt2=(end_x,end_y), color=COLOR, thickness=REC_THICKNESS)
    return new_img

def draw_center(image, center_x, center_y, COLOR):
    new_img = cv2.circle(image, (center_x, center_y), radius=5, color=COLOR, thickness=-1)
    return new_img

def draw_label(image, x, y, color, label):
    new_img = cv2.putText(image, label, (x,y), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)
    return new_img

def left_click(event, clicked_x, clicked_y, flags, params):

    global x, y, w, h, type
    if event == cv2.EVENT_LBUTTONUP and MODE != Tooth.NO_BOX:
        draw = True
        for index in range(len(x)):
            if clicked_x >= x[index] and clicked_x <= x[index]+w[index] and clicked_y >= y[index] and clicked_y <= y[index]+h[index]:
                x = np.delete(x, index)
                y = np.delete(y, index)
                w = np.delete(w, index)
                h = np.delete(h, index)
                type = np.delete(type, index)
                draw = False
                break

        if draw and (MODE == Tooth.CENTER_T or MODE == Tooth.CENTER_G):
            center_t = np.where(type == Tooth.CENTER_T)
            for index in center_t[0]:
                x = np.delete(x, index)
                y = np.delete(y, index)
                w = np.delete(w, index)
                h = np.delete(h, index)
                type = np.delete(type, index)

            center_g = np.where(type == Tooth.CENTER_G)
            for index in center_g[0]:
                x = np.delete(x, index)
                y = np.delete(y, index)
                w = np.delete(w, index)
                h = np.delete(h, index)
                type = np.delete(type, index)

        if draw:
            new_x = int(clicked_x - 1/2 * SQUARE)
            new_y = int(clicked_y - 1/2 * SQUARE)
            x = np.append(x, new_x)
            y = np.append(y, new_y)
            w = np.append(w, SQUARE)
            h = np.append(h, SQUARE)
            type = np.append(type, MODE)

        keyboard = Controller()
        keyboard.press("a")
        keyboard.release("a")

def save(file_name, img_name, mode):
    if mode == Match.ONE_D:  
        PATH_IMG = os.path.join("processed", "manual visualization 1D", file_name)
        PATH_DATA = os.path.join("processed", "manual data 1D", f"{img_name}.csv")
    else:
        PATH_IMG = os.path.join("processed", "manual visualization", file_name)
        PATH_DATA = os.path.join("processed", "manual data", f"{img_name}.csv")
    cv2.imwrite(PATH_IMG, clone)
    df = pd.DataFrame()
    df["x"] = x
    df["y"] = y
    df["w"] = w
    df["h"] = h
    df["type"] = type
    df.sort_values(by=["x"], inplace=True)
    df.to_csv(PATH_DATA)
    cv2.destroyAllWindows()

