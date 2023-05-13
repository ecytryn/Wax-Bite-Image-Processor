from enum import Enum
from dataclasses import dataclass
import os
import matplotlib.pyplot as plt
import cv2


# enum classes
class Match(Enum):
    TWO_D = 1
    ONE_D = 2

class Tooth(Enum):
    TOOTH = 1
    GAP = 2
    CENTER_T = 3
    CENTER_G = 4
    NO_BOX = 5

class Filter(Enum):
    GRADIENT = 1
    GRADIENT_EVEN = 2
    SMOOTH = 3
    SMOOTH_EVEN = 4
    NONE = 5


# settings/configurations of the program
@dataclass(frozen=True)
class CONFIG:
    "TEMPLATE MATCHING"
    #minimum score to be considered a tooth
    THRESHOLD = 0.75
    THRESHOLD_1D = 0.75
    #permited overlap to identify two "teeth" as distinct
    IOU_THRESHOLD = 0.05
    IOU_THRESHOLD_1D = 0.05   
    #methods to use for template matching; https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html for a list of methods
    METHODS = [cv2.TM_CCOEFF_NORMED] 

    "NOISE FILTERING"
    #threshold for gradient filtering
    GRAD_THRESHOLD = 5
    #threshold for gradient filtering (assuming teeth are equally spaced)
    GRAD_EVEN_THRESHOLD = 50
    #threshold for smoothness filtering
    SMOOTH_THRESHOLD = 0.5
    #threshold for smoothness filtering (assuming teeth are equally spaced)
    SMOOTH_EVEN_THRESHOLD = 5

    "HYPERBOLA SOLVE"
    #for intensity analysis; each data point is the average over a window WINDOW_WIDTH wide
    WINDOW_WIDTH = 10
    #which filtering technique to choose. See class Filter for options. 
    FILTER = Filter.NONE
    

    "OTHERS"
    # accepted filetypes to run analysis
    FILE_TYPES = [".jpg", ".png", ".jpeg"]
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
    DIRS_TO_MAKE = ['match visualization', 'match data', 'match visualization 1D', 'match data 1D',
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