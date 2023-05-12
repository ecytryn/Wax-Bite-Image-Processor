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

1. Solve 
2. Equidistant Set
3. Func
4. Plot Hyperbola Linear
'''

def solve(FILE_NAME, IMG_NAME, IMG_HEIGHT, FILTER, WINDOW_WIDTH: int = 0):

    current_dir = os.getcwd()
    os.chdir(os.path.join(current_dir,'processed', "filter data"))
    df = pd.read_csv(f"{IMG_NAME}_{FILTER}.csv")
    os.chdir(current_dir)

    x = df["x"].to_numpy()
    y = df["y"].to_numpy()

    matrix_t = [x**2, x*y, y**2, x, y]
    matrix = np.transpose(matrix_t)
    solved = np.matmul(np.linalg.inv(np.matmul(matrix_t, matrix)),np.matmul(matrix_t, np.ones(np.shape(matrix)[0])))
    (A,B,C,D,E) = solved
    
    error = False
    ends = np.roots([A, IMG_HEIGHT*B+D, -1+C*IMG_HEIGHT**2+E*IMG_HEIGHT])

    if B**2-4*A*C < 0:
        fit = plot_hyperbola_linear(min(ends), max(ends), solved)
        error = True
    else: 
        try: 
            fit = equidistant_set(min(ends), max(ends), solved)
        except RuntimeError as err:
            raise RuntimeError(err)

    img_path = os.path.join('img', FILE_NAME)
    img = cv2.imread(img_path)
    fig, ax = plt.subplots()
    ax.imshow(img, cmap=mpl.colormaps['gray'])
    
    ax.plot(fit[0], fit[1], '.-r', label="fit")
    target = os.path.join(current_dir,"processed", "fit visualization")
    os.chdir(target)
    fig.savefig(FILE_NAME)
    os.chdir(current_dir)

    if error:
        raise RuntimeError(f"Unable to fit a Hyperbola or Parabola; Circle or Ellipse detected.\nSee {FILE_NAME[0:len(FILE_NAME)-4]}.jpg in /processed/fit visualization for more detail.\nA={A}, B={B}, C={C}, D={D}, E={E}")

    projected_img = []
    normals_x = []
    normals_y = []

    for i in range(len(fit[0])):
        projection = project_1D.project_one(fit[0][i], fit[1][i], solved)
        temp = []
        normals_x.append(projection[2][0])
        normals_y.append(projection[2][1])
        for j in range(len(projection[0])):
            try:
                pixel = img[projection[1][j],projection[0][j]]
                temp.append([pixel[0], pixel[1], pixel[2]])
            except IndexError:
                temp.append([255,255,255])
        projected_img.append(temp)
        ax.plot(projection[0], projection[1], '.-y', label="projection")


    # sets up a data frame to store projection data
    df_project_data = pd.DataFrame()
    df_project_data["arclength loc"] = range(len(fit[0]))
    df_project_data["x_2D"] = fit[0]
    df_project_data["y_2D"] = fit[1]
    df_project_data["normal_x"] = normals_x
    df_project_data["normal_y"] = normals_y

    target = os.path.join(current_dir,"processed", "projection data", f"{IMG_NAME}.csv")
    df_project_data.to_csv(target)

    target = os.path.join(current_dir,"processed", "projection")
    os.chdir(target)
    projected_img_t = cv2.transpose(np.array(projected_img))
    cv2.imwrite(FILE_NAME, projected_img_t)
    os.chdir(current_dir)

    analyze_projection.avg_intesity(projected_img, WINDOW_WIDTH, FILE_NAME)

    target = os.path.join(current_dir,"processed", "projection sampling")
    os.chdir(target)
    fig.savefig(FILE_NAME)
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
    prev_x = start
    prev_y = start_y

    while prev_x < end:
        r1 = fsolve(func, np.pi/4, [prev_x, prev_y, coeff])
        r2 = fsolve(func, -np.pi/4, [prev_x, prev_y, coeff])

        if np.cos(r1[0])>0:
            curr_x, curr_y = prev_x+np.cos(r1[0]), prev_y+np.sin(r1[0])
        elif np.cos(r2[0])>0:
            curr_x, curr_y = prev_x+np.cos(r2[0]), prev_y+np.sin(r2[0])
        else:
            curr_1x, curr_1y = prev_x+np.cos(r1[0]), prev_y+np.sin(r1[0])
            curr_2x, curr_2y = prev_x+np.cos(r2[0]), prev_y+np.sin(r2[0])
            assert r1[0] < np.pi/2 and r1[0] > -np.pi/2, f"Equidistant Points Error: r1x_0 = {prev_x}, r1y_0 = {prev_y}, r1x_1={curr_1x}, r1x_2={curr_1y}\nr2x_0 = {prev_x}, r2y_0 = {prev_y}, r2x_1={curr_2x}, r2x_2={curr_2y}\n(A,B,C,D,E) = {coeff}"

        result[0].append(curr_x)
        result[1].append(curr_y)
        prev_x = curr_x
        prev_y = curr_y

    return result
    

def func(t, args):
    (prev_x, prev_y, coeff) = args
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
