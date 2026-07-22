import os
from dataclasses import dataclass
from enum import Enum, unique

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
    NO_BITE = 8
    CENTER_N = 9


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
@dataclass
class CONFIG:
    "TEMPLATE MATCHING"

    THRESHOLD: float = 0.75  # minimum score to be considered a tooth
    IOU_THRESHOLD: float = 0.05  # permited overlap to identify two "teeth" as distinct
    METHODS = [
        cv2.TM_CCOEFF_NORMED
    ]  # methods to use for template matching; https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html for a list of methods
    "NOISE FILTERING"
    GRAD_THRESHOLD: float = 5  # threshold for gradient filtering
    GRAD_EVEN_THRESHOLD: float = (
        50  # threshold for gradient filtering (assuming teeth are equally spaced)
    )
    SMOOTH_THRESHOLD: float = 0.5  # threshold for smoothness filtering
    SMOOTH_EVEN_THRESHOLD: float = (
        5  # threshold for smoothness filtering (assuming teeth are equally spaced)
    )
    "HYPERBOLA SOLVE"
    FILTER: Filter = Filter.MANUAL  # which filtering technique to choose (changes which file to take data from). See class Filter for options.
    "ANALYZE PROJECTION"
    WINDOW_WIDTH: int = 10  # for intensity analysis; each data point is the average over a window WINDOW_WIDTH wide
    ERROR_LOWER_B: int = 30  # lower distance bound for potential error
    ERROR_UPPER_B: int = 60  # upper distance bound for potential error
    "CROSS PROD"
    CROSS_METHOD: Cross = Cross.SQAURED  # which sums method to use
    CROSS: int = 3  # how many pairs of teeth around the current tooth to cross
    "PROJECT 1D"
    SAMPLING_WIDTH: int = 100  # how far away to sample from hyperbola for projection

    "GUI"
    SQUARE: int = 30  # side length of default GUI squares
    MAX_WIDTH: int | None = (
        1200  # max width of window in pixels; increase if image appears too small, decrease if it doesn't fit on screen
    )

    "FORMAT_PLOT"
    RESULT_PATH = os.path.join(
        os.getcwd(), "processed", "manual"
    )  # path to plot results from; this folder will be seen to contain the result data
    DATA_FILENAME = "manual data 1D.csv"  # file name of data (will be multiple named this within the output path)

    "OTHERS - STYLISTIC"
    # colors for manual editing; in format (G,B,R) not (R,G,B)!
    CENTER: tuple[int, int, int] = (255, 255, 0)  # cyan
    GAP: tuple[int, int, int] = (0, 255, 255)  # yellow
    TOOTH: tuple[int, int, int] = (0, 0, 255)  # red
    ERROR: tuple[int, int, int] = (0, 165, 255)  # orange
    NO_BITE: tuple[int, int, int] = (180, 130, 70)  # steel blue
    PLOT_STYLE: str = "default"  # plot style used by matplotlib
    # matplotlib figure dimensions (used when output is too crammed)
    WIDTH_SIZE: int = 15
    HEIGHT_SIZE: int = 7
    "OTHERS - INITIALIZATION"
    FILE_TYPES = [
        ".jpg",
        ".png",
        ".jpeg",
    ]  # accepted filetypes for templates and images


def apply_custom_width(width: int) -> None:
    CONFIG.MAX_WIDTH = width
