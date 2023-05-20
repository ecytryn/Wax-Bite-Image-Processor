# library imports
import os 
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
from scipy.optimize import fsolve


#helper functions
from cross_prod_center import crossProdCenter
import GUI
from utils import CONFIG, Filter, Match
import project_1D
import analyze_projection

'''
This file contains:
1. Solve - fits a hyperbola to the filtered data and performs projection
2. Equidistant Set - returns a set of coordinates on the hyperbola equidistant in terms of arclength
3. Func - helper for Equidistant Set; non-linear system to solve to obtain equidistant data points
4. Plot Hyperbola Linear - returns a set of coordinates on the hyperbola equidistant in terms of x
'''

def solve(fileName, imgName, fileType, imgHeight):

    currDir = os.getcwd()
    if CONFIG.FILTER == Filter.GRADIENT:
        df = pd.read_csv(os.path.join(currDir,'processed', "filter", imgName, "grad filtered.csv"))
    elif CONFIG.FILTER == Filter.GRADIENT_EVEN:
        df = pd.read_csv(os.path.join(currDir,'processed', "filter", imgName, "grad even filtered.csv"))
    elif CONFIG.FILTER == Filter.SMOOTH:
        df = pd.read_csv(os.path.join(currDir,'processed', "filter", imgName, "smooth filtered.csv"))
    elif CONFIG.FILTER == Filter.SMOOTH_EVEN:
        df = pd.read_csv(os.path.join(currDir,'processed', "filter", imgName, "smooth even filtered.csv"))
    else:
        df = pd.read_csv(os.path.join(currDir,'processed', "filter", imgName, "raw.csv"))

    x = df["x"].to_numpy()
    y = df["y"].to_numpy()

    matrixT = [x**2, x*y, y**2, x, y]
    matrix = np.transpose(matrixT)
    coeff = np.matmul(np.linalg.inv(np.matmul(matrixT, matrix)),np.matmul(matrixT, np.ones(np.shape(matrix)[0])))

    (A,B,C,D,E) = coeff

    error = False
    ends = np.roots([A, imgHeight*B+D, -1+C*imgHeight**2+E*imgHeight])

    if B**2-4*A*C < 0:
        fit = plotHyperbolaLinear(min(ends), max(ends), coeff)
        error = True
    else: 
        try: 
            fit = equidistantSet(min(ends), max(ends), coeff)
        except RuntimeError as err:
            raise RuntimeError(err)

    imgPath = os.path.join('img', fileName)
    img = cv2.imread(imgPath)
    fig, ax = plt.subplots()
    ax.imshow(img, cmap=mpl.colormaps['gray'])
    ax.plot(fit[0], fit[1], '.-r', label="fit")

    target = os.path.join(currDir,"processed", "fit", imgName)
    os.chdir(target)
    fig.savefig(f"fit.{fileType}")
    os.chdir(currDir)

    if error:
        raise RuntimeError(f"Unable to fit a Hyperbola or Parabola; Circle or Ellipse detected.\nSee 'fit.{fileType}' in /processed/fit/{imgName} for more detail.\nA={A}, B={B}, C={C}, D={D}, E={E}")

    projectedImg = []
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
        projectedImg.append(temp)
        ax.plot(projection[0], projection[1], '.-y', label="projection")

    # sets up a data frame to store projection data
    dfProjData = pd.DataFrame()
    dfProjData["arclength loc"] = range(len(fit[0]))
    dfProjData["x2D"] = fit[0]
    dfProjData["y2D"] = fit[1]
    dfProjData["tangentX"] = tangentsX
    dfProjData["tangentY"] = tangentsY
    dfProjData["normalX"] = normalsX
    dfProjData["normalY"] = normalsY


    target = os.path.join(currDir,"processed", "projection", imgName)
    os.chdir(target)
    dfProjData.to_csv(f"projection.csv")
    projectedImgT = cv2.transpose(np.array(projectedImg))
    cv2.imwrite(f"projection.{fileType}", projectedImgT)

    # intensity analysis; uncomment to perform
    analyze_projection.avgIntensity(imgName, fileType, projectedImg)

    fig.savefig(f"projection sampling.{fileType}")
    os.chdir(currDir)


    if CONFIG.FILTER == Filter.MANUAL:
        
        teethDf = pd.DataFrame()
        closestProjIndeces = []
        closestYs = []
        side = [50 for _ in range(len(x))]

        for toothInd in range(len(x)):
            projData = project_1D.projectData(x[toothInd], y[toothInd],coeff) #return (x, distance)
            closestX = projData[0] 
            closestY = projData[1]
            closestProjIndeces.append(np.argmin([abs(i-closestX) for i in fit[0]]))
            closestYs.append(CONFIG.SAMPLING_WIDTH+closestY)

        teethDf["x"] = closestProjIndeces
        teethDf["y"] = closestYs
        teethDf["w"] = side
        teethDf["h"] = side
        dfManual = pd.read_csv(os.path.join("processed", "manual", imgName, f"manual data.csv"))
        centerInd = crossProdCenter(coeff, df)

        t = dfManual.index[dfManual["type"]=="Tooth.CENTER_T"].to_numpy()
        g = dfManual.index[dfManual["type"]=="Tooth.CENTER_G"].to_numpy()
        assert len(t)+len(g) < 2, f"more than one center tooth or gap found in {imgName}.csv"
        # if center tooth doesn't exist 
        if len(t)+len(g) == 0:
            if dfManual["type"][centerInd] == "Tooth.TOOTH":
                dfManual["type"][centerInd] = "Tooth.CENTER_T"
            elif dfManual["type"][centerInd] == "Tooth.GAP":
                dfManual["type"][centerInd] = "Tooth.CENTER_G"
            dfManual.to_csv(os.path.join("processed", "manual", imgName, f"manual data.csv"))
        else:
            if len(t) > 0 and centerInd != t[0] or len(g) > 0 and centerInd != g[0]:
                    print(f"Alternative center index found: {centerInd}; please ensure the current center index is correct")
        teethDf["type"] = dfManual["type"]
        
        GUI.plotPreviousData(fileName, imgName, fileType, Match.ONE_D, teethDf)


def equidistantSet(start, end, coeff):

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


def plotHyperbolaLinear(start, end, coeff):
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
