import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import pandas as pd

from utils import Match, CONFIG, suffix

def templateMatching(self, mode: Match) -> None:
    """Performs template matching on an ImageProcessor object with templates in "template". 
    Saves numerical and visual results to /processed/template matching 
    
    Params
    ------
    self: ImageProcessor object
    mode: one of Match.TWO_D or Match_ONE_D depending on which to match

    Notes
    -----
    Useful Links:
    https://docs.opencv.org/4.x/d4/dc6/tutorial_py_templateMatching.html (a tutorial for template matching)
    https://docs.opencv.org/4.x/df/dfb/group__imgproc__object.html#gga3a7850640f1fe1f58fe91a2d7583695da5be00b45a4d99b5e42625b4400bfde65 (equations for each algorithm)
    """

    # set config thresholds, image data, template path
    if mode == Match.TWO_D:
        threshold = CONFIG.THRESHOLD
        iouThreshold = CONFIG.IOU_THRESHOLD
        templates = [file for file in os.listdir(self._PATH_TEMPLATE) if suffix(file) in CONFIG.FILE_TYPES]
        img = self.image
        template_path = self._PATH_TEMPLATE
    elif mode == Match.ONE_D:
        threshold = CONFIG.THRESHOLD_1D
        iouThreshold = CONFIG.IOU_THRESHOLD_1D
        templates = [file for file in os.listdir(self._PATH_TEMPLATE_1D) if suffix(file) in CONFIG.FILE_TYPES]
        img = self.image_proj
        template_path = self._PATH_TEMPLATE_1D

    teeth = []
    for template in templates:
        # load template and dimensions
        t = cv2.imread(os.path.join(template_path, template),cv2.IMREAD_GRAYSCALE)
        template_h, template_w = t.shape

        for method in CONFIG.METHODS:
            img_clone = img.copy()
            matching_score = cv2.matchTemplate(img_clone, t, method)
            #returns locations where matching_score is bigger than THRESHOLD
            filtered_matches = np.where(matching_score >= threshold)

            # for each (x,y)
            for pt in zip(*filtered_matches[::-1]):
                intersect = False
                for tooth in teeth[::]:
                    if intersectionOverUnion([pt[0], pt[1], template_w, template_h,
                                              matching_score[pt[1]][pt[0]]], tooth) > iouThreshold:
                        # if a location that intersects has a better matching score, replace
                        if matching_score[pt[1]][pt[0]] > tooth[4]:
                            teeth.remove(tooth)
                        else: 
                            intersect = True
                
                # if no intersection, add to list of teeth
                if not intersect:
                    newTooth = [pt[0], pt[1], template_w, template_h, matching_score[pt[1]][pt[0]], template]
                    teeth.append(newTooth)
        

    csv_data = {'x':[],'y':[], 'w':[],'h':[], 'score':[], 'match':[]}

    # for each identified tooth, draw a rectangle
    matched_image = img.copy()

    for pt in teeth:
        cv2.rectangle(matched_image, (pt[0], pt[1]), (pt[0] + pt[2], pt[1] + pt[3]), (255,255,0), 2)
        csv_data['x'].append(pt[0])
        csv_data['y'].append(pt[1])
        csv_data['w'].append(CONFIG.SQUARE)
        csv_data['h'].append(CONFIG.SQUARE)
        csv_data['score'].append(pt[4])
        csv_data['match'].append(pt[5])
    df = pd.DataFrame(data=csv_data)
    df.sort_values(by=['x'], inplace=True)

    # saving
    os.chdir(self._PATH_MATCHING)
    if mode == Match.TWO_D:
        df.to_csv("template matching.csv")
        cv2.imwrite(f"template matching{self.file_type}", matched_image)
    elif mode == Match.ONE_D:
        df.to_csv("template matching 1D.csv")
        cv2.imwrite(f"template matching 1D{self.file_type}", matched_image)
    os.chdir(self._PATH_ROOT)



def intersectionOverUnion(p1: list[int, int, int, int], p2: list[int, int, int, int]) -> float:
    """
    returns the intersection area over the union area of two template matches (intersection
    over union score)

    Params
    ------
    p1: [x, y, w, h] (box 1)
    p2: [x, y, w, h] (box 2)
    """
    #calculation of overlap; A = topleft corner, B = bottomright corner
    x_top_left = max(p1[0], p2[0])
    y_top_left = max(p1[1], p2[1])
    x_bot_right = min(p1[0]+p1[2], p2[0]+p2[2])
    y_bot_right = min(p1[1]+p1[3], p2[1]+p2[3])
    inter_area = max(0, x_bot_right - x_top_left + 1) * max(0, y_bot_right - y_top_left + 1)
    box1_area = p1[2] * p1[3]
    box2_area = p2[2] * p2[3]

    # score of overlap
    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou 