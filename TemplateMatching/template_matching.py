import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import pandas as pd
import time

# THRESHOLDs (tweek these!)
THRESHOLD = 0.8
IOU_THRESHOLD = 0.1
FILETYPE = ".jpg"
METHODS = [cv2.TM_CCOEFF_NORMED] 

def template_matching(IMG_NAME, TEMPLATES, FILE_TYPE):
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
    teeth = []
    for template in TEMPLATES:
        start_time = time.time()
        # load template
        t = cv2.imread(os.path.join("template", template),cv2.IMREAD_GRAYSCALE)

        # load images and dimensions 
        img = cv2.imread(os.path.join("img", IMG_NAME), cv2.IMREAD_GRAYSCALE)
        h, w = t.shape

        for method in METHODS:
            img2 = img.copy()
            result = cv2.matchTemplate(img2, t, method)

            #returns locations where result is bigger than THRESHOLD
            loc = np.where(result >= THRESHOLD)

            # for each (x,y)
            for pt in zip(*loc[::-1]):
                intersect = False
                for tooth in teeth[::]:
                    if intersection_over_union([pt[0], pt[1],w,h,result[pt[1]][pt[0]]], tooth) > IOU_THRESHOLD:
                        # if a location that intersects has a better matching score, replace
                        if result[pt[1]][pt[0]] > tooth[4]:
                            teeth.remove(tooth)
                        else: 
                            intersect = True
                
                # if no intersection, add to list of teeth
                if not intersect:
                    new_tooth = [pt[0], pt[1], w, h, result[pt[1]][pt[0]]]
                    teeth.append(new_tooth)
        

    data = {'x':[],'y':[], 'w':[],'h':[], 'score':[]}
    img = cv2.imread(os.path.join("img", IMG_NAME), cv2.IMREAD_GRAYSCALE)
    for pt in teeth:
        cv2.rectangle(img, (pt[0], pt[1]), (pt[0] + pt[2], pt[1] + pt[3]), (255,255,0), 2)
        data['x'].append(pt[0])
        data['y'].append(pt[1])
        data['w'].append(pt[2])
        data['h'].append(pt[3])
        data['score'].append(pt[4])
    df = pd.DataFrame(data=data)
    df.to_csv(f"{IMG_NAME}_processed.csv")
    # plt.scatter(data['x'], data['y'])
    # plt.show()
    cv2.imwrite(f"{IMG_NAME}{template[:len(template)-4]}_processed{FILE_TYPE}", img)


def intersection_over_union(p1, p2):
    """
    outputs the intersection area size over the union area size of two boxes. p1 and p2 are the top left
    coordinates of the boxes. 
    """
    #calculation of overlap
    xA = max(p1[0], p2[0])
    yA = max(p1[1], p2[1])
    xB = min(p1[0]+p1[2], p2[0]+p2[2])
    yB = min(p1[1]+p1[3], p2[1]+p2[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    box1Area = p1[2] * p1[3]
    box2Area = p2[2] * p2[3]

    # score of overlap
    iou = interArea / float(box1Area + box2Area - interArea)
    return iou 




if __name__ == "__main__":
    images = [file for file in os.listdir(os.path.join(os.getcwd(),"img")) if file[len(file)-4:] == FILETYPE]
    templates = [file for file in os.listdir(os.path.join(os.getcwd(),"template")) if file[len(file)-4:] == FILETYPE]
    for image in images:
        template_matching(image, templates, FILETYPE)




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


# df = pd.read_csv("01_05_2021.jpg_processed.csv")
# data = []
# for i in range(df.shape[0]):
#     data.append([df['x'][i], df['y'][i], df['w'][i], df['h'][i], df['score'][i]])

# for i in data:
#     for j in data:
#         if (intersection_over_union(i, j) > IOU_THRESHOLD and i != j):
#             print("point 1", i)
#             print("point 2", j)
#             print("\n")