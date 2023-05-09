import os 
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import project_1D
import analyze_projection
from scipy.optimize import fsolve

'''
This file contains:




'''

def solve(CSV, IMG_TYPE, IMAGE_HEIGHT, filter, intensity_window_width: int = 0):

    current_dir = os.getcwd()
    os.chdir(os.path.join(current_dir,'processed', "filter data"))
    df = pd.read_csv(f"{CSV[:len(CSV)-4]}_{filter}.csv")
    os.chdir(current_dir)

    x = df["x"].to_numpy()
    y = df["y"].to_numpy()

    matrix_t = [x**2, x*y, y**2, x, y]
    matrix = np.transpose(matrix_t)
    solved = np.matmul(np.linalg.inv(np.matmul(matrix_t, matrix)),np.matmul(matrix_t, np.ones(np.shape(matrix)[0])))
    (A,B,C,D,E) = solved
    
    error = False
    ends = np.roots([A, IMAGE_HEIGHT*B+D, -1+C*IMAGE_HEIGHT**2+E*IMAGE_HEIGHT])

    if B**2-4*A*C < 0:
        fit = plot_hyperbola_linear(min(ends), max(ends), solved)
        error = True
    else: 
        try: 
            fit = equidistant_set(min(ends), max(ends), solved)
        except RuntimeError as err:
            raise RuntimeError(err)

    img_path = os.path.join('img', f"{CSV[:len(CSV)-4]}{IMG_TYPE}")
    img = cv2.imread(img_path)
    fig, ax = plt.subplots()
    ax.imshow(img, cmap=mpl.colormaps['gray'])
    
    ax.plot(fit[0], fit[1], '.-r', label="fit")
    target = os.path.join(current_dir,"processed", "fit visualization")
    os.chdir(target)
    fig.savefig(f"{CSV[0:len(CSV)-4]}.jpg")
    os.chdir(current_dir)

    if error:
        raise RuntimeError(f"Unable to fit a Hyperbola or Parabola; Circle or Ellipse detected.\nSee {CSV[0:len(CSV)-4]}.jpg in /processed/fit visualization for more detail.\nA={A}, B={B}, C={C}, D={D}, E={E}")

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

    target = os.path.join(current_dir,"processed", "projection")
    os.chdir(target)
    cv2.imwrite(f"{CSV[0:len(CSV)-4]}.jpg", np.array(projected_img))
    os.chdir(current_dir)

    analyze_projection.avg_intesity(projected_img, intensity_window_width, CSV)

    target = os.path.join(current_dir,"processed", "projection sampling")
    os.chdir(target)
    fig.savefig(f"{CSV[0:len(CSV)-4]}.jpg")
    os.chdir(current_dir)


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



    result = ([],[])
    prev_root = 0
    prev_x = start
    prev_y = start_y

    for i in range(100):
        base_root = fsolve(func, i)
        check = fsolve(func, base_root, prev_x, prev_y)
        if (check[0] < base_root[0]):
            prev_root = base_root[0] + np.pi
            break
        elif i == 99:
            raise RuntimeError(f"Suitable equidistant arclength points cannot be found")

    while prev_x < end:
        root = fsolve(func, prev_root, prev_x, prev_y, coeff)
        prev_root = root
        rx, ry = prev_x+np.cos(root[0]), prev_y+np.sin(root[0])

        result[0].append(rx)
        result[1].append(ry)
        prev_x = rx
        prev_y = ry
    
    return result
    

def func(t, prev_x, prev_y, coeff):
    (A,B,C,D,E) = coeff
    return A*(prev_x+np.cos(t))**2+B*(prev_x+np.cos(t))*(prev_y+np.sin(t))+C*(prev_y+np.sin(t))**2+D*(prev_x+np.cos(t))+E*(prev_y+np.sin(t))-1


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
