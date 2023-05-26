import cv2
import os
import pandas as pd
import numpy as np
from pynput.keyboard import Key, Controller
from utils import Tooth, Match, CONFIG


class GUI:
    """
    An image editing interface to identify teeth, gaps, center teeth, center gaps of a teeth wax image. 

    Usage:

    MOUSE
    -----
    left-click: identify an item of the current mode (tooth, gap, center tooth, or center gap)
    left-click(inside box): remove an item

    KEYBOARD
    --------
    tab: switch between modes in order
    1: enter "tooth" mode
    2: enter "gap" mode
    3: enter "center tooth" mode
    4. enter "center gap" mode
    space: show or hide boxes
    save: save data and exit
    esc: exit without saving
    """

    # tooth, gap, center tooth, center gap: 4 modes
    MODES = [mode for mode in Tooth if mode != Tooth.NO_BOX]

    def __init__(self, file_name: str, img_name: str, file_type: str, matching_mode: Match) -> None:
        """
        Initializes an instance of image editing interface.

        If mode is Match.ONE_D, read 
            1) image from /processed/projection/[img_name]/projection[filetype];
            2) data from /processed/manual/[img_name]/manual data 1D.csv
                2.1) if doesn't exist, read /processed/template matching/[img_name]/template matching 1D.csv
                2.2) if doesn't exist, use empty data set

        If mode is Match.TWO_D, read 
            1) image from /img/[file_name]
            2) data from /processed/manual/[img_name]/manual data.csv
                2.1) if doesn't exist, read /processed/template matching/[img_name]/template matching.csv
                2.2) if doesn't exist, use empty data set

        Params
        ------
        file_name: image name with file extension
        img_name: image name without file extension
        file_type: image extensiom
        mode: one of Match.ONE_D or Match.TWO_D 
        """

        self.x = np.array([])
        self.y = np.array([])
        self.w = np.array([])
        self.h = np.array([])
        self.type = np.array([])
        self._stored_mode = Tooth.TOOTH
        self._curr_mode = Tooth.TOOTH

        self.file_name = file_name
        self.img_name = img_name
        self.file_type = file_type
        self.matching_mode = matching_mode

        # find data path
        if matching_mode == Match.ONE_D:
            img_path = os.path.join("processed", "projection", img_name, f"projection{file_type}")
            manual_data_path = os.path.join("processed", "manual", img_name, "manual data 1D.csv")

            if os.path.isfile(manual_data_path):
                img_data_path = manual_data_path
            else:
                img_data_path = os.path.join("processed", "template matching", img_name, "template matching 1D.csv")
        else:
            img_path = os.path.join("img", file_name)
            manual_data_path = os.path.join("processed", "manual", img_name, f"manual data.csv")

            if os.path.isfile(manual_data_path):
                img_data_path = manual_data_path
            else:
                img_data_path = os.path.join("processed", "template matching", img_name, f"template matching.csv") 


        # if image doesn't exist, raise error and stop
        if not os.path.isfile(img_path):
            raise RuntimeError(f"{img_path} does not exist. Image deleted or a hyperbola fit was likely not found: did you run fitProject first?")
        
        # if data doesn't exist, use empty set, otherwise use existing data
        if not os.path.isfile(img_data_path):
            print(f"{img_data_path} not found, empty data set used")
        else:
            # load data
            df = pd.read_csv(img_data_path)
            self.x = df["x"].to_numpy()
            self.y = df["y"].to_numpy()
            self.w = df["w"].to_numpy()
            self.h = df["h"].to_numpy()

            # if type column isn't in data (possible for filtered image), assume every data point is a tooth
            if "type" not in df.columns:
                df["type"] = [Tooth.TOOTH for _ in range(len(df.index))]
                self.type = df["type"]
            else: 
                for i in df["type"]:
                    if i == "Tooth.TOOTH":
                        self.type = np.append(self.type, Tooth.TOOTH)
                    elif i == "Tooth.GAP":
                        self.type = np.append(self.type, Tooth.GAP)
                    elif i == "Tooth.CENTER_T":
                        self.type = np.append(self.type, Tooth.CENTER_T)
                    elif i == "Tooth.CENTER_G":
                        self.type = np.append(self.type, Tooth.CENTER_G)
                    elif i == "Tooth.ERROR_T":
                        self.type = np.append(self.type, Tooth.ERROR_T)
                    elif i == "Tooth.ERROR_G":
                        self.type = np.append(self.type, Tooth.ERROR_G)

        # reads image, set up mouse callback
        image = cv2.imread(img_path)
        cv2.namedWindow(img_name)
        cv2.setMouseCallback(img_name, self.left_click)
        self.mode_index = 0

        while True:

            # makes a copy of the original image 
            self.clone = image.copy()
            dataset_size = len(self.x)

            # draw data
            for i in range(dataset_size):
                self.clone = GUI.draw_tooth(self.clone, 
                                            self.x[i], 
                                            self.y[i], 
                                            self.w[i], 
                                            self.h[i], 
                                            self.type[i], 
                                            self._curr_mode)

            # sets current mode and display image
            if self._curr_mode == Tooth.TOOTH:
                text_img = cv2.putText(self.clone, "mode: tooth", (10,40), cv2.FONT_HERSHEY_DUPLEX, 1.5, CONFIG.TOOTH, 2)
            elif self._curr_mode == Tooth.GAP: 
                text_img = cv2.putText(self.clone, "mode: gap", (10,40), cv2.FONT_HERSHEY_DUPLEX, 1.5, CONFIG.GAP, 2)
            elif self._curr_mode == Tooth.CENTER_T: 
                text_img = cv2.putText(self.clone, "mode: center tooth", (10,40), cv2.FONT_HERSHEY_DUPLEX, 1.5, CONFIG.CENTER, 2)
            elif self._curr_mode == Tooth.CENTER_G:
                text_img = cv2.putText(self.clone, "mode: center gap", (10,40), cv2.FONT_HERSHEY_DUPLEX, 1.5, CONFIG.CENTER, 2)
            else:
                text_img = cv2.putText(self.clone, "mode: viewing", (10,40), cv2.FONT_HERSHEY_DUPLEX, 1.5, (100,100,100), 2)

            # resize window size
            self.ratio = CONFIG.MAX_WIDTH / text_img.shape[1]
            dim = (CONFIG.MAX_WIDTH, int(text_img.shape[0] * self.ratio))
            resized = cv2.resize(text_img, dim, interpolation=cv2.INTER_AREA)
            cv2.imshow(img_name, resized)

            break_loop = self.wait_keyboard_logic()
            if break_loop:
                break


    def wait_keyboard_logic(self) -> None:
        """
        defines keyboard logic and waits for one keypress
        """
        key = cv2.waitKey(0)

        if key == 32: # space
            # if in viewing mode already, store mode 
            if self._curr_mode == Tooth.NO_BOX:
                self._curr_mode = self._stored_mode
            else:
                # else store current mode, change into viewing mode
                self._stored_mode = self._curr_mode
                self._curr_mode = Tooth.NO_BOX
        elif key == 27: # esc
            cv2.destroyAllWindows()
            return True
        elif key == 9 and self._curr_mode != Tooth.NO_BOX: # tab
            # change into the next mode
            self.mode_index = (self.mode_index + 1) % len(self.MODES)
            self._curr_mode = self.MODES[self.mode_index]
        elif key == ord("s"): # s
            df_res = pd.DataFrame()
            df_res["x"] = self.x
            df_res["y"] = self.y
            df_res["w"] = self.w
            df_res["h"] = self.h
            df_res["type"] = self.type
            df_res.sort_values(by=["x"], inplace=True)
            GUI.save(self.file_name, self.img_name, self.file_type, self.matching_mode, self.clone, df_res)
            return True
        elif key == ord("1"): # 1
            self._curr_mode = Tooth.TOOTH
            self.mode_index = 0
        elif key == ord("2"): # 2
            self._curr_mode = Tooth.GAP
            self.mode_index = 1
        elif key == ord("3"): # 3
            self._curr_mode = Tooth.CENTER_T
            self.mode_index = 2
        elif key == ord("4"): # 4
            self._curr_mode = Tooth.CENTER_G
            self.mode_index = 3
        return False
    


    @staticmethod
    def draw_tooth(image: list[list[list[int]]], x: float, y: float, w: float, 
                  h: float, type: Tooth, curr_mode: Tooth | None = None) -> list[list[list[int]]]:
        """
        Draws a type of element at a location on an image

        Params
        ------
        image: image
        x: x coordinate of top left location of data point
        y: y coordinate of top left location of data point
        w: width of data point
        h: height of data point
        type: one of Tooth.TOOTH, Tooth.GAP, Tooth.CENTER_T, Tooth.CENTER_G
        curr_mode: if curr mode is Tooth.NO_BOX, don't draw box

        Returns
        -------
        altered image
        """
        # offset for T and G in center tooth box (added so it's centered)
        OFFSET_X = -5
        OFFSET_Y = 5

        center_x = int(x + 1/2 * w)
        center_y = int(y + 1/2 * h)
        end_x = x + w
        end_y = y + h

        if type == Tooth.TOOTH:
            color = CONFIG.TOOTH
        elif type == Tooth.GAP:
            color = CONFIG.GAP
        elif type == Tooth.CENTER_T:
            color = CONFIG.CENTER
            labelled_img = GUI._draw_label(image, center_x+OFFSET_X, center_y+OFFSET_Y, color, "T")
        elif type == Tooth.CENTER_G:
            color = CONFIG.CENTER
            labelled_img = GUI._draw_label(image, center_x+OFFSET_X, center_y+OFFSET_Y, color, "G")
        elif type == Tooth.ERROR_T:
            color = CONFIG.ERROR
            labelled_img = GUI._draw_label(image, center_x+OFFSET_X, center_y+OFFSET_Y, color, "T")
        elif type == Tooth.ERROR_G:
            color = CONFIG.ERROR
            labelled_img = GUI._draw_label(image, center_x+OFFSET_X, center_y+OFFSET_Y, color, "G")

        labelled_img = GUI._draw_center(image, center_x, center_y, color)

        # if no box, return 
        if curr_mode == Tooth.NO_BOX:
            return labelled_img
        # draw box
        image_rec = GUI._draw_rectangle(labelled_img, x, y, end_x, end_y, color)
        return image_rec
    
    @staticmethod
    def _draw_rectangle(image: list[list[list[int]]], x: int, y: int, 
                       end_x: int, end_y: int, color: tuple[int, int, int]) -> list[list[list[int]]]:
        """
        Draws a rectangle at a specified location on an image

        Params
        ------
        image: image
        x: x coordinate of top left location of rectangle
        y: y coordinate of top left location of rectangle
        end_x: x coordinate of bottom right location of rectangle
        end_y: y coordinate of bottom right location of rectangle
        color: color of rectangle

        Returns
        -------
        altered image
        """

        new_image = cv2.rectangle(image, pt1=(x,y), pt2=(end_x,end_y), color=color, thickness=2)
        return new_image
    
    @staticmethod
    def _draw_center(image: list[list[list[int]]], center_x: int, center_y: int, 
                    color: tuple[int, int, int]):
        """
        Draws a point at a specified location on an image

        Params
        ------
        image: image
        center_x: x coordinate of point
        center_y: y coordinate of point
        color: color of rectangle

        Returns
        -------
        altered image
        """
        new_image = cv2.circle(image, (center_x, center_y), radius=5, color=color, thickness=-1)
        return new_image
    
    @staticmethod
    def _draw_label(image: list[list[list[int]]], x: int, y: int, 
                   color: tuple[int, int, int], label: str) -> list[list[list[int]]]:
        """
        Draws text at a specified location on an image

        Params
        ------
        image: image
        x: x coordinate of text
        y: y coordinate of text
        color: color of rectangle
        label: text to draw

        Returns
        -------
        altered image
        """
        new_image = cv2.putText(image, label, (x,y), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)
        return new_image


    def left_click(self, event, clicked_x: float, clicked_y: float, flags, params) -> None:
        """
        Event handler for mouse click. Registers the location when released.

        - If in viewing mode: clicking doesn't do anything
        - If a box is clicked on: box deleted
        - If self._curr_mode is Tooth.CENTER_T or Tooth.CENTER_G: clicking another location 
        will remove the previous center since there can only be one center
        - Otherwise: draws the element specified by self._curr_mode at the location of mouse click
        """
        if event == cv2.EVENT_LBUTTONUP and self._curr_mode != Tooth.NO_BOX:
            draw = True
            dataset_size = len(self.x)

            clicked_x = clicked_x / self.ratio
            clicked_y = clicked_y / self.ratio

            for index in range(dataset_size):
                # if the user clicks within a square, delete the square 
                if (clicked_x >= self.x[index] and clicked_x <= self.x[index]+self.w[index] 
                    and clicked_y >= self.y[index] and clicked_y <= self.y[index]+self.h[index]):
                    self._delete_index_data(index)
                    draw = False
                    break
            
            # if the user clicks a new center, delete the old one
            if draw and (self._curr_mode == Tooth.CENTER_T or self._curr_mode == Tooth.CENTER_G):
                center_t = np.where(self.type == Tooth.CENTER_T)
                for index in center_t[0]:
                    self._delete_index_data(index)
                center_g = np.where(self.type == Tooth.CENTER_G)
                for index in center_g[0]:
                    self._delete_index_data(index)

            # otherwise, draw a square at the clicked location
            if draw:
                new_x = int(clicked_x - 1/2 * CONFIG.SQUARE)
                new_y = int(clicked_y - 1/2 * CONFIG.SQUARE)
                self.x = np.append(self.x, new_x)
                self.y = np.append(self.y, new_y)
                self.w = np.append(self.w, CONFIG.SQUARE)
                self.h = np.append(self.h, CONFIG.SQUARE)
                self.type = np.append(self.type, self._curr_mode)

            # this section tricks the GUI into reloading by pressing "a"
            keyboard = Controller()
            keyboard.press("a")
            keyboard.release("a")

    @staticmethod
    def save(file_name: str, img_name: str, file_type: str, mode: Match, image: list[list[list[int]]], 
             df_res: pd.DataFrame) -> None:
        """
        Save image with edited data. 

        If mode is Match.ONE_D
            1) saves image to /processed/manual/[img_name]/manual 1D[file_type]
            1) saves data to /processed/manual/[img_name]/manual data 1D.csv
        If mode is Match.TWO_D: 
            1) saves image to /processed/manual/[img_name]/manual[file_type]
            1) saves data to /processed/manual/[img_name]/manual data.csv

        Params
        ------
        file_name: name of image with file extension (not needed but for uniform purposes)
        img_name: name of image without file extension
        file_extension: file extension of image
        mode: one of Match.ONE_D or Match.TWO_D
        image: image to save
        """

        if mode == Match.ONE_D:  
            PATH_IMG = os.path.join("processed", "manual", img_name, f"manual 1D{file_type}")
            PATH_DATA = os.path.join("processed", "manual", img_name, f"manual data 1D.csv")
        else:
            PATH_IMG = os.path.join("processed", "manual", img_name, f"manual{file_type}")
            PATH_DATA = os.path.join("processed", "manual", img_name, f"manual data.csv")

        cv2.imwrite(PATH_IMG, image)
        df_res.to_csv(PATH_DATA)
        cv2.destroyAllWindows()

    def _delete_index_data(self, index: int) -> None:
        """
        """
        self.x = np.delete(self.x, index)
        self.y = np.delete(self.y, index)
        self.w = np.delete(self.w, index)
        self.h = np.delete(self.h, index)
        self.type = np.delete(self.type, index)
