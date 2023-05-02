import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread("img/sep_15_2017 LG_219_U_0.8x.jpg", cv2.IMREAD_GRAYSCALE)
template = cv2.imread("template/1.jpg",cv2.IMREAD_GRAYSCALE)
h, w = template.shape

# methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR, 
#            cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]

methods = [cv2.TM_CCOEFF_NORMED]

for method in methods:
    img2 = img.copy()
    result = cv2.matchTemplate(img2, template, method)

    # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    # if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
    #     location = min_loc
    # else:
    #     location = max_loc
    # bottom_right = (location[0] + w, location[1] + h)


    threshold = 0.5
    iou_threshold = 0.3
    loc = np.where(result >= threshold) #returns indices where this happens
    teeth = []

    for pt in zip(*loc[::-1]):
        intersect = False
        for tooth in teeth:
            xA = max(pt[0], tooth[0])
            yA = max(pt[1], tooth[1])
            xB = min(pt[0]+w, tooth[0]+w)
            yB = min(pt[1]+h, tooth[1]+h)
            interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

            iou = interArea / float(2*w*h - interArea)
            if iou > iou_threshold:
                intersect = True
                if result[pt[1]][pt[0]] > result[tooth[1]][tooth[0]]:
                    tooth = pt
                    
        if not intersect:
            teeth.append(pt)
            cv2.rectangle(img2, pt, (pt[0] + w, pt[1] + h), (255,255,0), 2)

    plt.subplot(121),plt.imshow(result,cmap = 'gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img2,cmap = 'gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(method)
    plt.show()
