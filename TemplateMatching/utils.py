from enum import Enum, unique
from dataclasses import dataclass
import os
import matplotlib.pyplot as plt
import cv2
from typing import Tuple
import datetime


# enum classes
@unique
class Match(Enum):
    # types of matches 
    TWO_D = 1
    ONE_D = 2

@unique
class Tooth(Enum):
    # types of data
    TOOTH = 1
    GAP = 2
    CENTER_T = 3
    CENTER_G = 4
    NO_BOX = 5

@unique
class Filter(Enum):
    # types of filter
    GRADIENT = 1
    GRADIENT_EVEN = 2
    SMOOTH = 3
    SMOOTH_EVEN = 4
    NONE = 5
    MANUAL = 6

@unique
class Cross(Enum):
    # types of method for cross product step
    SQAURED = 1
    ABS = 2


# settings/configurations of the program
@dataclass(frozen=True)
class CONFIG:
    "TEMPLATE MATCHING"
    #minimum score to be considered a tooth
    THRESHOLD: float = 0.75
    THRESHOLD_1D: float = 0.75
    #permited overlap to identify two "teeth" as distinct
    IOU_THRESHOLD: float = 0.05
    IOU_THRESHOLD_1D: float = 0.05   
    #methods to use for template matching; https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html for a list of methods
    METHODS = [cv2.TM_CCOEFF_NORMED] 

    "NOISE FILTERING"
    #threshold for gradient filtering
    GRAD_THRESHOLD: float = 5
    #threshold for gradient filtering (assuming teeth are equally spaced)
    GRAD_EVEN_THRESHOLD: float = 50
    #threshold for smoothness filtering
    SMOOTH_THRESHOLD: float = 0.5
    #threshold for smoothness filtering (assuming teeth are equally spaced)
    SMOOTH_EVEN_THRESHOLD: float = 5

    "HYPERBOLA SOLVE"
    #for intensity analysis; each data point is the average over a window WINDOW_WIDTH wide
    WINDOW_WIDTH: int = 10
    #which filtering technique to choose. See class Filter for options. 
    FILTER: Filter = Filter.MANUAL
    
    "CROSS PROD"
    # which cross product method to use
    CROSS_METHOD: Cross = Cross.SQAURED
    # how many pairs of teeth around the current tooth to cross
    CROSS: int = 3

    "PROJECT 1D"
    # how far away to sample from hyperbola for projection
    SAMPLING_WIDTH: int = 100

    "GUI"
    # side length of default manual squares
    SQUARE = 30

    "PLOT_MANUAL"
    # how much "time" elapsed between each image
    TIME = 2
    # PATH to plot results from; this folder will be seen to contain the result data
    PATH = os.path.join(os.getcwd(),"processed", "manual")

    "OTHERS - STYLISTIC"
    #colors for manual editing; in format (G,B,R) not (R,G,B)!
    CENTER: Tuple[int, int, int] = (255,255,0) #cyan
    GAP: Tuple[int, int, int] = (0,255,255) #yellow
    TOOTH: Tuple[int, int, int] = (0,0,255) #red
    # plot style used by matplotlib
    PLOT_STYLE: str = "bmh"
    #matplotlib figure dimensions (used when output is too crammed)
    WIDTH_SIZE: int = 15
    HEIGHT_SIZE: int = 7

    "OTHERS - INITIALIZATION"
    # accepted filetypes for templates and images
    FILE_TYPES = [".jpg", ".png", ".jpeg"]


# general helper functions
def makeDir(dir: str):
    '''create a specified directory if it doesn't already exist'''
    if not os.path.isdir(dir):
        os.mkdir(dir)

def suffix(file: str):
    '''returns the suffix of a file'''
    return os.path.splitext(file)[1]

def endProcedure():
    '''closes all current matplotlib and cv2 windows'''
    plt.close("all")
    cv2.destroyAllWindows()

def printDivider():
    '''prints a divider into the console'''
    print("============================================================")


def parseDate(fileName):
    '''parses the date from the name of an image (assumes format YYYY_MM_DD...)'''
    try:
        year = int(fileName[0:4])
        month = int(fileName[5:7])
        day = int(fileName[8:10])
    except Exception as e:
        print(f"filename {fileName} is not in the correct format")

    return datetime.datetime(year, month, day)
    