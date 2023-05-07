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

    normal_x = []
    normal_y = []

    c = 0
    while True:
        distance = math.sqrt((c*normal[0])**2+(c*normal[1])**2)
        if distance < 50:
            normal_x.append(int(x+c*normal[0]))
            normal_y.append(int(y+c*normal[1]))
            normal_x.append(int(x-c*normal[0]))
            normal_y.append(int(y-c*normal[1]))
            c += 1
        else:
            break
    
    df = pd.DataFrame({'x': normal_x, 'y': normal_y})
    df.sort_values(by=['x'], inplace=True)
    return (df['x'].to_numpy(), df['y'].to_numpy())

# if __name__ == "__main__":
#     data = [file for file in os.listdir(os.getcwd()) if file[len(file)-4:] == ".csv"]
#     project(data[0])