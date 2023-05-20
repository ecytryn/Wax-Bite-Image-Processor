import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from scipy.signal import find_peaks
from utils import CONFIG

def avgIntensity(imgName, fileType, data):
    currDir = os.getcwd()
    fig2, ax2 = plt.subplots()

    fig2.set_figwidth(CONFIG.WIDTH_SIZE)
    fig2.set_figheight(CONFIG.HEIGHT_SIZE)
    fig2.tight_layout()

    avgIntensity = np.mean(data, axis=1)
    avgWindowIntensity = []
    for i in range(len(avgIntensity)):
        if i - int(CONFIG.WINDOW_WIDTH/2) < 0:
            start = 0
        else: 
            start = i - int(CONFIG.WINDOW_WIDTH/2)
        if i + int(CONFIG.WINDOW_WIDTH/2) > len(data):
            end = len(data)
        else: 
            end = i + int(CONFIG.WINDOW_WIDTH/2)
        avgWindowIntensity.append(np.mean(avgIntensity[start:end], axis=0))
    
    avgIntensityGraph = [255-i[0] for i in avgWindowIntensity]

    # assumes image is at current dir
    projection = cv2.imread(f"projection.{fileType}")

    ax2.imshow(projection)
    ax2.plot(range(len(avgIntensityGraph)),avgIntensityGraph, color='y')
    localMaxIndex, _ = find_peaks(avgIntensityGraph, distance=30)
    ax2.scatter(localMaxIndex,np.ones(len(localMaxIndex)), color='r')


    fig2.savefig(f"projection intensity.{fileType}")
    os.chdir(currDir)
