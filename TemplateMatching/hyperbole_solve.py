import os 
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


def solve():
    pass



if __name__ == "__main__":
    data = [file for file in os.listdir(os.getcwd()) if file[len(file)-4:] == ".csv"]
    continuity_filter(data)