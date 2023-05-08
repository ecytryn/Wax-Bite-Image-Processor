import os 
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import hyperbola_solve
import math


def project(csv):
    coeff = hyperbola_solve.solve(csv)
    ((A, B, C, D, E), start, end, x,y) = coeff
    fig, ax = plt.subplots()
    # plt.imshow(img, cmap=mpl.colormaps['gray'])
    ax.plot(x, y, '.-r', label="fit")
    plt.show()
    print(A)


def project_one(x, y, coeff):
    (A, B, C, D, E) = coeff
    

    tangent = (1, -B/C+1/(2*math.sqrt(((B*x+E)/C)**2-4*((A*x**2+D*x-1)/C)))*(2*(B*x+E)/C*B/C-4/C*(2*A*x+D)))
    normal = (1, -1/(tangent[1]))
    normal_h = (normal[0]/math.sqrt((normal[0])**2+(normal[1])**2), normal[1]/math.sqrt((normal[0])**2+(normal[1])**2))

    normal_x = []
    normal_y = []

    c = 50
    while c >= -50:
        normal_x.append(int(x+c*normal_h[0]))
        normal_y.append(int(y+c*normal_h[1]))
        c -= 1
    
    df = pd.DataFrame({'x': normal_x, 'y': normal_y})
    df.sort_values(by=['y'], inplace=True)
    return (df['x'].to_numpy(), df['y'].to_numpy())

# if __name__ == "__main__":
#     data = [file for file in os.listdir(os.getcwd()) if file[len(file)-4:] == ".csv"]
#     project(data[0])