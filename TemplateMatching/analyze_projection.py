import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from scipy.signal import find_peaks

def avg_intesity(data, window_width, CSV):
    """
    
    """
    WIDTH_SIZE = 15
    HEIGHT_SIZE = 7
    current_dir = os.getcwd()
    fig2, ax2 = plt.subplots()

    fig2.set_figwidth(WIDTH_SIZE)
    fig2.set_figheight(HEIGHT_SIZE)
    fig2.tight_layout()

    avg_intensity = np.mean(data, axis=1)
    avg_window_intensity = []
    for i in range(len(avg_intensity)):
        if i - int(window_width/2) < 0:
            start = 0
        else: 
            start = i - int(window_width/2)
        if i + int(window_width/2) > len(data):
            end = len(data)
        else: 
            end = i + int(window_width/2)
        avg_window_intensity.append(np.mean(avg_intensity[start:end], axis=0))
    
    avg_intensity_graph = [255-i[0] for i in avg_window_intensity]
    
    target = os.path.join(current_dir,"processed", "projection")
    os.chdir(target)
    projection = cv2.imread(f"{CSV[0:len(CSV)-4]}.jpg")
    projection = cv2.transpose(projection)
    os.chdir(current_dir)

    ax2.imshow(projection)
    ax2.plot(range(len(avg_intensity_graph)),avg_intensity_graph, color='y')

    local_max_index, _ = find_peaks(avg_intensity_graph, distance=30)
    ax2.scatter(local_max_index,np.ones(len(local_max_index)), color='r')


    target = os.path.join(current_dir,"processed", "projection graphed")
    os.chdir(target)
    fig2.savefig(f"{CSV[0:len(CSV)-4]}.jpg")
    os.chdir(current_dir)
