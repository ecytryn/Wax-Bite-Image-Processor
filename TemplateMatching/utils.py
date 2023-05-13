from enum import Enum
from dataclasses import dataclass
import os
import matplotlib.pyplot as plt
import cv2


class Match(Enum):
    TWO_D = 1
    ONE_D = 2

class Tooth(Enum):
    TOOTH = 1
    GAP = 2
    CENTER_T = 3
    CENTER_G = 4
    NO_BOX = 5


@dataclass(frozen=True)
class PARAMS:
    "NOISE FILTERING"
    

    "OTHERS"
    #colors for manual editing; in format (G,B,R) not (R,G,B)!
    CENTER = (255,255,0) #cyan
    GAP=(0,255,255) #yellow
    TOOTH=(0,0,255) #red

    # plot style used by matplotlib
    PLOT_STYLE = "bmh"

    #matplotlib figure dimensions (used when output is too crammed)
    WIDTH_SIZE = 15
    HEIGHT_SIZE = 7

    # directories that will be created in /processed
    DIRS_TO_MAKE = ['match visualization', 'match data', 'match visualization 1D', 'match data 1D'
                'filter visualization', 'filter data', 'fit visualization',
                'projection', 'projection sampling', 'projection graphed', 'projection data',
                'manual data', 'manual visualization', 'manual data 1D', 'manual visualization 1D']


# helper functions
def make_dir(dir: str):
    '''create a specified directory if it doesn't already exist'''
    if not os.path.isdir(dir):
        os.mkdir(dir)

def suffix(file: str):
    '''returns the suffix of a file'''
    return os.path.splitext(file)[1]

def end_procedure():
    '''closes all current matplotlib and cv2 windows'''
    plt.close("all")
    cv2.destroyAllWindows()

def print_divider():
    '''prints a divider into the console'''
    print("============================================================")