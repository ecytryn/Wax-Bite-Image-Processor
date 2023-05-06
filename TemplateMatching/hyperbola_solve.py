import os 
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import noise_filtering


def solve(data):
    for csv in data: 
        filtered = noise_filtering.continuity_filter(csv)
        x = filtered[0].to_numpy()
        y = filtered[1].to_numpy()
        matrix_t = [x**2, x*y, y**2, x, y]
        matrix = np.transpose(matrix_t)
        solved = np.matmul(np.linalg.inv(np.matmul(matrix_t, matrix)),np.matmul(matrix_t, np.ones(np.shape(matrix)[0])))
        fit = plot_hyperbola(x[0], x[-1], solved)

        fig, ax = plt.subplots()
        ax.plot(fit[0], fit[1], '.-y', label="fit")
        ax.plot(x, y, '.-', label="data")
        plt.show()

def plot_hyperbola(start, end, coeff):
    x = np.linspace(start, end, num=100)
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
    solve(data[0:1])