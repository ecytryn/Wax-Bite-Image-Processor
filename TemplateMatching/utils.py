from enum import Enum, unique
from dataclasses import dataclass
import os
import cv2


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
    ERROR_T = 6
    ERROR_G = 7

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
    #which filtering technique to choose (changes which file to take data from). See class Filter for options. 
    FILTER: Filter = Filter.MANUAL

    "ANALYZE PROJECTION"
    #for intensity analysis; each data point is the average over a window WINDOW_WIDTH wide
    WINDOW_WIDTH: int = 10
    ERROR_LOWER_B: int = 30
    ERROR_UPPER_B: int = 60


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
    SQUARE: int = 30
    MAX_WIDTH: int | None = None # shouldn't be large than image width

    "PLOT_MANUAL"
    # how much "time" elapsed between each image
    TIME = 2
    # PATH to plot results from; this folder will be seen to contain the result data
    PATH = os.path.join(os.getcwd(),"processed", "manual")
    DATA_FILENAME = "manual data 1D.csv"

    "OTHERS - STYLISTIC"
    #colors for manual editing; in format (G,B,R) not (R,G,B)!
    CENTER: tuple[int, int, int] = (255, 255, 0) #cyan
    GAP: tuple[int, int, int] = (0, 255, 255) #yellow
    TOOTH: tuple[int, int, int] = (0, 0, 255) #red
    ERROR: tuple[int, int, int] = (0, 165, 255) #orange
    # plot style used by matplotlib
    PLOT_STYLE: str = "default"
    #matplotlib figure dimensions (used when output is too crammed)
    WIDTH_SIZE: int = 15
    HEIGHT_SIZE: int = 7

    "OTHERS - INITIALIZATION"
    # accepted filetypes for templates and images
    FILE_TYPES = [".jpg", ".png", ".jpeg"]
    