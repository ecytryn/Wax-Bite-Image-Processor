import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread("img/aug_4_2017 LG_219_U_0.8xtry4.jpg", cv2.IMREAD_GRAYSCALE)
template = cv2.imread("template/1.jpg",cv2.IMREAD_GRAYSCALE)
h, w = template.shape

methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR, 
           cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]

for method in methods:
    img2 = img.copy()
    result = cv2.matchTemplate(img2, template, method)


    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        location = min_loc
    else:
        location = max_loc
    bottom_right = (location[0] + w, location[1] + h)

    # cv2.imshow("Match", img2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    cv2.rectangle(img2, location, bottom_right, 255, 2)
    plt.subplot(121),plt.imshow(result,cmap = 'gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img,cmap = 'gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(method)
    plt.show()