import os 
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import noise_filtering
import cv2
import project_1D
from PIL import Image



def solve(csv):
    filtered = noise_filtering.continuity_filter(csv)
    x = filtered[0].to_numpy()
    y = filtered[1].to_numpy()
    matrix_t = [x**2, x*y, y**2, x, y]
    matrix = np.transpose(matrix_t)
    solved = np.matmul(np.linalg.inv(np.matmul(matrix_t, matrix)),np.matmul(matrix_t, np.ones(np.shape(matrix)[0])))
    fit = plot_hyperbola(x[0], x[-1], solved)

    img = cv2.imread(f'{csv[:len(csv)-4]}.jpg')
    img = cv2.imread(f'test.jpg')
    fig, ax = plt.subplots()
    plt.imshow(img, cmap=mpl.colormaps['gray'])
    
    projected_img = []
    for i in range(len(fit[0])):
        projection = project_1D.project_one(fit[0][i], fit[1][i], solved)
        temp = []
        for j in range(len(projection[0])):
            pixel = img[projection[1][j],projection[0][j]]
            temp.append([pixel[0], pixel[1], pixel[2]])
        
        ax.plot(projection[0], projection[1], '.-y', label="projection")

        #padding 
        missing = 100-len(temp)
        for _ in range(missing): 
            temp.append([255,255,255])
        projected_img.append(temp)

    ax.plot(fit[0], fit[1], '.-r', label="fit")
    plt.savefig(f"{csv[0:len(csv)-4]}_fitted.png")
    cv2.imwrite("test.png", np.array(projected_img))
    # plt.show()
    # return(solved, x[0], x[-1], fit[0], fit[1])

def plot_hyperbola(start, end, coeff):
    x = np.linspace(start, end, num=int(end-start)+1)
    quadratic = coeff[2]*np.ones(len(x))
    linear = coeff[1]*x+coeff[4]
    constant = coeff[0]*x**2+coeff[3]*x-1

    result = ([],[])
    for i in range(len(x)):
        roots = np.roots([quadratic[i], linear[i], constant[i]])
        for r in roots:
            if r >= 0:
                result[0].append(x[i])
                result[1].append(r)
    return result
    
if __name__ == "__main__":
    data = [file for file in os.listdir(os.getcwd()) if file[len(file)-4:] == ".csv"]
    for csv in data[0:1]:
        solve(csv)
