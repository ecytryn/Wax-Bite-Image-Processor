import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import pandas as pd

def template_matching(IMG_NAME, TEMPLATE_NAME, FILE_TYPE):
    """ This function reads all jpg from the img folder and runs multi-template matching on it.
    The "coordinates" for teeth (top-left pixel of it) is outputted in a CSV file. Copies of the 
    images labelled with the suspected teeth are also generated for reference.
    
    Note:
    1. This function detects objects similar in size to the template provided. It does not scale the template
    in any way and find matches that way. Please ensure that the target object is similar in size to how it 
    appears in the image
    2. Possible algorithms for template matching: [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR, 
    cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]. Please note that the current code is taylored 
    to cv2.TM_CCOEFF_NORMED
    3. The folder structure needed for this code to run is an img and template folder in the same directory,
    where the img folder has all the images

    Useful Links:
    https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html (a tutorial for template matching)
    https://docs.opencv.org/4.x/df/dfb/group__imgproc__object.html#gga3a7850640f1fe1f58fe91a2d7583695da5be00b45a4d99b5e42625b4400bfde65 (equations for each algorithm)
    """

    # thresholds (tweek these!)
    threshold = 0.5
    iou_threshold = 0.1

    # load template
    template = cv2.imread(os.path.join("template", TEMPLATE_NAME),cv2.IMREAD_GRAYSCALE)

    # load images and dimensions 
    img = cv2.imread(os.path.join("img", IMG_NAME), cv2.IMREAD_GRAYSCALE)
    h, w = template.shape

    #methods of choice
    methods = [cv2.TM_CCOEFF_NORMED] 

    for method in methods:
        
        img2 = img.copy()
        result = cv2.matchTemplate(img2, template, method)

        #returns locations where result is bigger than threshold
        loc = np.where(result >= threshold)
        teeth = []

        # for each (x,y)
        for pt in zip(*loc[::-1]):
            intersect = False
            for tooth in teeth:
                
                #calculation for overlap
                xA = max(pt[0], tooth[0])
                yA = max(pt[1], tooth[1])
                xB = min(pt[0]+w, tooth[0]+w)
                yB = min(pt[1]+h, tooth[1]+h)
                interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

                # score of overlap
                iou = interArea / float(2*w*h - interArea)
                if iou > iou_threshold:
                    intersect = True
                    # if a location that intersects has a better matching score, replace
                    if result[pt[1]][pt[0]] > result[tooth[1]][tooth[0]]:
                        tooth = pt
            
            # if no intersection, add to list of teeth
            if not intersect:
                teeth.append(pt)
        

        data = {'x':[],'y':[]}
        for pt in teeth:
            cv2.rectangle(img2, pt, (pt[0] + w, pt[1] + h), (255,255,0), 2)
            data['x'].append(pt[0])
            data['y'].append(pt[1])

        df = pd.DataFrame(data=data)
        df.to_csv(f"{IMG_NAME}_processed.csv")
        # plt.scatter(data['x'], data['y'])
        # plt.show()

        cv2.imwrite(f"{IMG_NAME}_processed{FILE_TYPE}", img2)



if __name__ == "__main__":
    FILETYPE = ".jpg"
    TEMPLATE = "1.jpg"
    directory_items = os.listdir(os.path.join(os.getcwd(),"img"))
    
    for item in directory_items:
        if item[len(item)-4:] == ".jpg":
            template_matching(item, TEMPLATE,FILETYPE)

# code for single template matching
    # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    # if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
    #     location = min_loc
    # else:
    #     location = max_loc
    # bottom_right = (location[0] + w, location[1] + h)

# code for plotting
    #plt.subplot(121),plt.imshow(result,cmap = 'gray')
    #plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    #plt.subplot(122),plt.imshow(img2,cmap = 'gray')
    #plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    #plt.suptitle(method)
    #plt.show()