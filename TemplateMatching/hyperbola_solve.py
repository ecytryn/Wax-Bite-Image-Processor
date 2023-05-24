# library imports
import os 
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
from scipy.optimize import fsolve


#helper functions
from sum_dot_prod import sum_dot_prod
import GUI
from utils import CONFIG, Filter, Match
import project_1D
import analyze_projection


def solve(file_name: str, img_name: str, file_type: str, img_height: int):

    # load the correct filtered data
    curr_dir = os.getcwd()
    if CONFIG.FILTER == Filter.GRADIENT:
        df = pd.read_csv(os.path.join(curr_dir,'processed', "filter", img_name, "grad filtered.csv"))
    elif CONFIG.FILTER == Filter.GRADIENT_EVEN:
        df = pd.read_csv(os.path.join(curr_dir,'processed', "filter", img_name, "grad even filtered.csv"))
    elif CONFIG.FILTER == Filter.SMOOTH:
        df = pd.read_csv(os.path.join(curr_dir,'processed', "filter", img_name, "smooth filtered.csv"))
    elif CONFIG.FILTER == Filter.SMOOTH_EVEN:
        df = pd.read_csv(os.path.join(curr_dir,'processed', "filter", img_name, "smooth even filtered.csv"))
    else:
        df = pd.read_csv(os.path.join(curr_dir,'processed', "filter", img_name, "raw.csv"))

    x = df["x"].to_numpy()
    y = df["y"].to_numpy()

    # matrix operations to solve for coefficient of best fitting conic; also solves for x-intercepts (bottom edge of image)
    matrixT = [x**2, x*y, y**2, x, y]
    matrix = np.transpose(matrixT)
    coeff = np.matmul(np.linalg.inv(np.matmul(matrixT, matrix)),np.matmul(matrixT, np.ones(np.shape(matrix)[0])))
    (A,B,C,D,E) = coeff
    x_inter = np.roots([A, img_height*B+D, -1+C*img_height**2+E*img_height])

    # if conic is a hyperbola, find equidistant set
    error = False
    if B**2-4*A*C < 0:
        fit = plot_hyperbola_linear(min(x_inter), max(x_inter), coeff)
        error = True
    else: 
        try: 
            fit = equidistant_set(min(x_inter), max(x_inter), coeff)
        except RuntimeError as err:
            raise RuntimeError(err)

    # plot fit
    img_path = os.path.join('img', file_name)
    img = cv2.imread(img_path)
    fig, ax = plt.subplots()
    ax.imshow(img, cmap=mpl.colormaps['gray'])
    ax.plot(fit[0], fit[1], '.-r', label="fit")
    # save figure 
    target = os.path.join(curr_dir,"processed", "fit", img_name)
    os.chdir(target)
    fig.savefig(f"fit{file_type}")
    os.chdir(curr_dir)
    
    # if conic not a hyperbola, raise error
    if error:
        raise RuntimeError(f"Unable to fit a Hyperbola or Parabola; Circle or Ellipse detected.\nSee 'fit{file_type}' in /processed/fit/{img_name} for more detail.\nA={A}, B={B}, C={C}, D={D}, E={E}")

    proj_img = []
    normalsX = []
    normalsY = []
    tangentsX =[]
    tangentsY = []

    for i in range(len(fit[0])):
        projection = project_1D.projectOne(fit[0][i], fit[1][i], coeff)
        temp = []
        normalsX.append(projection[2][0])
        normalsY.append(projection[2][1])
        tangentsX.append(projection[3][0])
        tangentsY.append(projection[3][1])
        for j in range(len(projection[0])):
            try:
                pixel = img[projection[1][j],projection[0][j]]
                temp.append([pixel[0], pixel[1], pixel[2]])
            except IndexError:
                temp.append([255,255,255])
        proj_img.append(temp)
        ax.plot(projection[0], projection[1], '.-y', label="projection")

    # sets up a data frame to store projection data
    df_proj = pd.DataFrame()
    df_proj["arclength loc"] = range(len(fit[0]))
    df_proj["x2D"] = fit[0]
    df_proj["y2D"] = fit[1]
    df_proj["tangentX"] = tangentsX
    df_proj["tangentY"] = tangentsY
    df_proj["normalX"] = normalsX
    df_proj["normalY"] = normalsY


    target = os.path.join(curr_dir,"processed", "projection", img_name)
    os.chdir(target)
    df_proj.to_csv(f"projection.csv")
    proj_img_transpose = cv2.transpose(np.array(proj_img))
    cv2.imwrite(f"projection{file_type}", proj_img_transpose)
    fig.savefig(f"projection sampling{file_type}")
    os.chdir(curr_dir)

    analyze_projection.avg_intensity(img_name, file_type, proj_img)


    if CONFIG.FILTER == Filter.MANUAL:
        
        teeth_df = pd.DataFrame()
        closest_proj_indecies = []
        closest_ys = []
        side = [50 for _ in range(len(x))]

        for tooth_ind in range(len(x)):
            projData = project_1D.projectData(x[tooth_ind], y[tooth_ind],coeff) #return (x, distance)
            closest_x = projData[0] 
            closest_y = projData[1]
            closest_proj_indecies.append(np.argmin([abs(i-closest_x) for i in fit[0]]))
            closest_ys.append(CONFIG.SAMPLING_WIDTH+closest_y)

        teeth_df["x"] = closest_proj_indecies
        teeth_df["y"] = closest_ys
        teeth_df["w"] = side
        teeth_df["h"] = side
        df_manual = pd.read_csv(os.path.join("processed", "manual", img_name, f"manual data.csv"))
        center_ind = sum_dot_prod(coeff, df)

        t = df_manual.index[df_manual["type"]=="Tooth.CENTER_T"].to_numpy()
        g = df_manual.index[df_manual["type"]=="Tooth.CENTER_G"].to_numpy()
        assert len(t)+len(g) < 2, f"more than one center tooth or gap found in {img_name}.csv"

        # if center tooth doesn't exist 
        if len(t)+len(g) == 0:
            if df_manual["type"][center_ind] == "Tooth.TOOTH":
                df_manual["type"][center_ind] = "Tooth.CENTER_T"
            elif df_manual["type"][center_ind] == "Tooth.GAP":
                df_manual["type"][center_ind] = "Tooth.CENTER_G"
        else:
            if len(t) > 0 and center_ind != t[0] or len(g) > 0 and center_ind != g[0]:
                    print(f"Alternative center index found: {center_ind}; please ensure the current center index is correct")


        # find and mark potential errors
        potential_errors = analyze_projection.arclength_histogram(img_name, file_type, closest_proj_indecies)
        for e in potential_errors:
            if df_manual["type"][e] == "Tooth.TOOTH":
                df_manual["type"][e] = "Tooth.ERROR_T"
            elif df_manual["type"][e] == "Tooth.GAP":
                df_manual["type"][e] = "Tooth.ERROR_G"

        # save and re-plot altered data
        df_manual.to_csv(os.path.join("processed", "manual", img_name, f"manual data.csv"))
        GUI.plot_previous_data(file_name, img_name, file_type, Match.TWO_D, df_manual)

        teeth_df["type"] = df_manual["type"]
        GUI.plot_previous_data(file_name, img_name, file_type, Match.ONE_D, teeth_df)
        


def equidistant_set(start, end, coeff):

    # equidistant in x
    x = np.linspace(start, end, num=int(end-start)+1)
    quadratic = coeff[2]*np.ones(len(x))
    linear = coeff[1]*x+coeff[4]
    constant = coeff[0]*x**2+coeff[3]*x-1

    #conic: Ax**2+Bxy+Cy**2+Dx+Ey-1=0
    #circle parameterization: x = prevX + cos(t); y = prevY + sin(t)
    #intersection: plug

    startRoots = [r for r in np.roots([quadratic[0], linear[0], constant[0]]) if r >= 0]
    startY = min(startRoots)

    result = ([],[])
    prevX = start
    prevY = startY

    while prevX < end:
        r1 = fsolve(func, np.pi/4, [prevX, prevY, coeff])
        r2 = fsolve(func, -np.pi/4, [prevX, prevY, coeff])

        if np.cos(r1[0])>0:
            currX, currY = prevX+np.cos(r1[0]), prevY+np.sin(r1[0])
        elif np.cos(r2[0])>0:
            currX, currY = prevX+np.cos(r2[0]), prevY+np.sin(r2[0])
        else:
            curr1x, curr1y = prevX+np.cos(r1[0]), prevY+np.sin(r1[0])
            curr2x, curr2y = prevX+np.cos(r2[0]), prevY+np.sin(r2[0])
            if r1[0] < np.pi/2 and r1[0] > -np.pi/2:
                raise RuntimeError(f"Equidistant Points Error: r1x_0 = {prevX}, r1y_0 = {prevY}, r1x_1={curr1x}, r1x_2={curr1y}\nr2x_0 = {prevX}, r2y_0 = {prevY}, r2x_1={curr2x}, r2x_2={curr2y}\n(A,B,C,D,E) = {coeff}\n Try readjusting some data through GUI")
        result[0].append(currX)
        result[1].append(currY)
        prevX = currX
        prevY = currY

    return result
    

def func(t, args):
    (prevX, prevY, coeff) = args
    (A,B,C,D,E) = coeff
    return A*(prevX+np.cos(t))**2+B*(prevX+np.cos(t))*(prevY+np.sin(t))+C*(prevY+np.sin(t))**2+D*(prevX+np.cos(t))+E*(prevY+np.sin(t))-1


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
