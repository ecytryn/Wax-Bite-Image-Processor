from enum import Enum

class Match(Enum):
    TWO_D = 1
    ONE_D = 2

class Tooth(Enum):
    TOOTH = 1
    GAP = 2
    CENTER_T = 3
    CENTER_G = 4
    NO_BOX = 5