import os 
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import noise_filtering
import cv2
import project_1D
from PIL import Image
from scipy.optimize import fsolve




def solve(csv):
    IMAGE_HEIGHT = 1944

    filtered = noise_filtering.continuity_filter(csv)
    x = filtered[0].to_numpy()
    y = filtered[1].to_numpy()
    matrix_t = [x**2, x*y, y**2, x, y]
    matrix = np.transpose(matrix_t)
    solved = np.matmul(np.linalg.inv(np.matmul(matrix_t, matrix)),np.matmul(matrix_t, np.ones(np.shape(matrix)[0])))
    (A,B,C,D,E) = solved
    ends = np.roots([A, IMAGE_HEIGHT*B+D, -1+C*IMAGE_HEIGHT**2+E*IMAGE_HEIGHT])
    print(ends)

    # fit = plot_hyperbola_linear(ends[0], ends[-1], solved)
    fit = equidistant_set(min(ends), max(ends), solved)
    
    # img = cv2.imread(f'{csv[:len(csv)-4]}.jpg')
    img = cv2.imread(f'test.jpg')
    fig, ax = plt.subplots()
    fig2, intensity_ax = plt.subplots()
    ax.imshow(img, cmap=mpl.colormaps['gray'])
    
    projected_img = []

    for i in range(len(fit[0])):
        projection = project_1D.project_one(fit[0][i], fit[1][i], solved)
        temp = []
        for j in range(len(projection[0])):
            try:
                pixel = img[projection[1][j],projection[0][j]]
                temp.append([pixel[0], pixel[1], pixel[2]])
            except IndexError:
                temp.append([255,255,255])
        projected_img.append(temp)
        ax.plot(projection[0], projection[1], '.-y', label="projection")


    ax.plot(fit[0], fit[1], '.-r', label="fit")

    gradient = np.mean(projected_img, axis=1)
    gradient_graph = [255-g[0] for g in gradient]
    gradient_map = []
    for _ in range(50):
        gradient_map.append(gradient)

    intensity_ax.bar(range(len(gradient_graph)),gradient_graph, color='k', align='center', width=1.0, label="fit")

    fig.savefig(f"{csv[0:len(csv)-4]}_fitted.png")
    fig2.savefig(f"gradient_graph.png")
    cv2.imwrite("test.png", np.array(projected_img))
    cv2.imwrite("gradient.png", np.array(gradient_map))
    # return(solved, x[0], x[-1], fit[0], fit[1])

def equidistant_set(start, end, coeff):

    # equidistant in x
    x = np.linspace(start, end, num=int(end-start)+1)
    quadratic = coeff[2]*np.ones(len(x))
    linear = coeff[1]*x+coeff[4]
    constant = coeff[0]*x**2+coeff[3]*x-1

    #Ax**2+Bxy+Cy**2+Dx+Ey-1=0
    #circle: x = prev_x + cos(t); y = prev_y + sin(t)
    #intersect: Ax**2+Bxy+Cy**2+Dx+Ey-1=0
    (A,B,C,D,E) = coeff

    start_roots = [r for r in np.roots([quadratic[0], linear[0], constant[0]]) if r >= 0]
    start_y = min(start_roots)

    prev_x = start
    prev_y = start_y

    def func(t):
        return A*(prev_x+np.cos(t))**2+B*(prev_x+np.cos(t))*(prev_y+np.sin(t))+C*(prev_y+np.sin(t))**2+D*(prev_x+np.cos(t))+E*(prev_y+np.sin(t))-1


    result = ([],[])
    prev_roots = [0, 0]
    while prev_x < end:
        roots = fsolve(func, prev_roots)
        prev_roots = roots
        assert len(roots) == 2, "more than 2 roots found for unit circle"
        r1x, r1y = prev_x+np.cos(roots[0]), prev_y+np.sin(roots[0])
        r2x, r2y = prev_x+np.cos(roots[1]), prev_y+np.sin(roots[1])
        print(r1x, r2x)
        if r1x > r2x:
            result[0].append(r1x)
            result[1].append(r1y)
            prev_x = r1x
            prev_y = r1y
        else: 
            result[0].append(r2x)
            result[1].append(r2y)
            prev_x = r2x
            prev_y = r2y

    return result
    

def plot_hyperbola_linear(start, end, coeff):
    # equidistant in x
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
