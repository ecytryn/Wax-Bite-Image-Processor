import os 
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


def continuity_filter(data):
    for csv in data:
        df = pd.read_csv(csv)
        x = df['x']
        y = df['y']
        plt.plot(x,y,'bo')
        plt.show()


def graph_filter():
    pass


if __name__ == "__main__":
    data = [file for file in os.listdir(os.getcwd()) if file[len(file)-4:] == ".csv"]
    continuity_filter(data)