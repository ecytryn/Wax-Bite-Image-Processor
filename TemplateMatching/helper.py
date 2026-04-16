import os
import matplotlib.pyplot as plt
import cv2
import datetime
import numpy as np
from scipy.optimize import fsolve
import pandas as pd

from utils import CONFIG


# ------------------------------------------------------------------------
"GENERAL"

def end_procedure() -> None:
    '''closes all current matplotlib and cv2 windows'''
    plt.close("all")
    cv2.destroyAllWindows()

def flag_to_integer(args: list[str], flag: str) -> int:
    """
    Checks whether the value followed by a flag is an integer. If
    yes, return the integer.

    Param
    -----
    args: list of command line arguments
    flag: flag after which to search for integer
    """
    value = args[args.index(flag)+1]
    try:
        index = int(value)
    except ValueError:
        raise ValueError(f"{value} is not an integer.")
    except IndexError:
        raise IndexError(f"No integer followed by {flag}")

    if index < 0:
        raise RuntimeError(f"Flag {flag} is non positive")
    return index

def make_dir(dir: str) -> None:
    '''create a specified directory if it doesn't already exist'''
    if not os.path.isdir(dir):
        os.mkdir(dir)

def print_divider() -> None:
    '''prints a divider into the console'''
    print("============================================================")

def parse_date(file_name: str) -> datetime.datetime:
    '''parses the date from the name of an image (assumes format MM_DD_YYYY or YYYY_MM_DD)'''
    try:
        if file_name[4] == '_':
            year = int(file_name[0:4])
            month = int(file_name[5:7])
            day = int(file_name[8:10])
        else:
            year = int(file_name[6:10])
            month = int(file_name[0:2])
            day = int(file_name[3:5])
    except Exception as e:
        print(f"filename {file_name} is not in the correct format (expected MM_DD_YYYY or YYYY_MM_DD)")
    return datetime.datetime(year, month, day)
    
def suffix(file: str) -> str:
    '''returns the suffix of a file'''
    return os.path.splitext(file)[1]


# ------------------------------------------------------------------------
"MATH"

def axis_symmetry(coeff: tuple[float, float, float, float, float]) -> np.array:
    """
    Compute the axis of symmetry that divides a hyperbola

    Params
    ------
    coeff: coefficients of hyperbola (A, B, C, D, E) for hyperbola having the 
    equation Ax^2 + Bxy + Cy^2 + Dx + Ey - 1 = 0

    Returns
    -------
    A vector parallel to the axis of symmetry 
    """
    (A,B,C,D,E) = coeff
    tan_twotheta = B/(A-C)
    theta = np.arctan(tan_twotheta)/2
    # sym_vector = (1, np.tan(theta)) # this one divides hyperbola into two, not bisect
    bisect_vector = np.array([np.tan(theta), -1])
    return bisect_vector


def equidistant_set(start: float, end: float, 
                    coeff: tuple[float, float, float, float, float]
                    ) -> tuple[np.array, np.array]:
    """
    Compute the coordinates on a hyperbola at every unit distanced location of the
    arclength from start to end.

    Params
    ------
    start: starting x
    end: ending x
    coeff: coefficients of hyperbola (A, B, C, D, E) for hyperbola having the 
    equation Ax^2 + Bxy + Cy^2 + Dx + Ey - 1 = 0

    Returns
    -------
    x, y coordinates of the equidistant data point
    """
    # points equidistant in x
    (A, B, C, D, E) = coeff
    quadratic_term = C
    linear_term = B*start + E
    constant_term = A*start**2 + D*start - 1

    # find the starting y
    start_root = [r for r in np.roots([quadratic_term, linear_term, constant_term]) if r >= 0]
    start_y = min(start_root)

    result = ([],[])
    prev_x = start
    prev_y = start_y

    while prev_x < end:
        r1 = fsolve(_equidistant_set_func, np.pi/4, [prev_x, prev_y, coeff])
        r2 = fsolve(_equidistant_set_func, -np.pi/4, [prev_x, prev_y, coeff])

        if np.cos(r1[0])>0:
            curr_x, curr_y = prev_x+np.cos(r1[0]), prev_y+np.sin(r1[0])
        elif np.cos(r2[0])>0:
            curr_x, curr_y = prev_x+np.cos(r2[0]), prev_y+np.sin(r2[0])
        else:
            curr1x, curr1y = prev_x+np.cos(r1[0]), prev_y+np.sin(r1[0])
            curr2x, curr2y = prev_x+np.cos(r2[0]), prev_y+np.sin(r2[0])
            if r1[0] < np.pi/2 and r1[0] > -np.pi/2:
                raise RuntimeError(f"""Equidistant Points Error: r1x_0 = {prev_x}, r1y_0 = {prev_y}, r1x_1={curr1x}, r1x_2={curr1y}\n
                r2x_0 = {prev_x}, r2y_0 = {prev_y}, r2x_1={curr2x}, r2x_2={curr2y}\n(A,B,C,D,E) = {coeff}\n Try readjusting some data through GUI""")
        result[0].append(curr_x)
        result[1].append(curr_y)
        prev_x = curr_x
        prev_y = curr_y

    return result


def _equidistant_set_func(t: float, 
                          args: tuple[float, tuple[float, float, float, float, float]]
                          ) -> float:
    """
    Used to solve for the next unit distant point away from the previous point

    Params
    ------
    t: guess for next unit distant point
    args: [prev_x, prev_y, coeff] where
        1. prev_x, prev_y = previus data point
        2. coeff = coefficients of hyperbola

    Returns
    -------
    Output when plugging t into the hyperbola equation. If the output is 0, it means that
    t yields a point on the hyperbola. 
    """
    (prev_x, prev_y, coeff) = args
    (A,B,C,D,E) = coeff

    return (A*(prev_x+np.cos(t))**2
            +B*(prev_x+np.cos(t))*(prev_y+np.sin(t))
            +C*(prev_y+np.sin(t))**2
            +D*(prev_x+np.cos(t))
            +E*(prev_y+np.sin(t))-1)

def intersection_over_union(p1: list[int, int, int, int], 
                            p2: list[int, int, int, int]
                            ) -> float:
    """
    Computes the intersection area over the union area of two boxes ('intersection
    over union' score). Helper of template_matching. 

    Params
    ------
    p1: [x, y, w, h] (box 1)
    p2: [x, y, w, h] (box 2)

    Returns
    -------
    iou: intersection over union score
    """
    #calculation of overlap; A = topleft corner, B = bottomright corner
    x_top_left = max(p1[0], p2[0])
    y_top_left = max(p1[1], p2[1])
    x_bot_right = min(p1[0]+p1[2], p2[0]+p2[2])
    y_bot_right = min(p1[1]+p1[3], p2[1]+p2[3])
    inter_area = max(0, x_bot_right - x_top_left + 1) * max(0, y_bot_right - y_top_left + 1)
    box1_area = p1[2] * p1[3]
    box2_area = p2[2] * p2[3]

    # score of overlap
    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou 

def plot_hyperbola_linear(start: float, end: float, 
                          coeff: tuple[float, float, float, float, float]) -> tuple[np.array, np.array]:
    """
    Compute the coordinates on a hyperbola for every integer x from start to end.

    Params
    ------
    start: starting x
    end: ending x
    coeff: coefficients of hyperbola (A, B, C, D, E) for hyperbola having the 
    equation Ax^2 + Bxy + Cy^2 + Dx + Ey - 1 = 0
    """

    (A, B, C, D, E) = coeff
    x = np.linspace(start, end, num=int(end-start)+1)
    quadratic = C * np.ones(len(x))
    linear = B*x + E
    constant = A*x**2 + D*x - 1
    result = ([],[])

    for coordinate_index in range(len(x)):
        roots = np.roots([quadratic[coordinate_index], 
                          linear[coordinate_index], 
                          constant[coordinate_index]])
        
        for r in roots:
            if r >= 0:
                result[0].append(x[coordinate_index])
                result[1].append(r)

    return result


def project_data_one(x: float, y: float, 
                     coeff: tuple[float, float, float, float, float]) -> tuple[float, float]:
    """
    Compute the x coordinate of the shortest distance from a point to a hyperbola
    with coeffcicients = coeff. Also compute the distance from the hyperbola at the 
    closest point

    Params
    ------
    x: x-coordinate of the point of projection 
    y: y-coordinate of the point of projection 
    coeff: coefficients of hyperbola (A, B, C, D, E) for hyperbola having the 
    equation Ax^2 + Bxy + Cy^2 + Dx + Ey - 1 = 0

    Returns
    -------
    1. arclength position of the point on the hyperbola closests to data point
    2. distance from data point to the closest point on the hyperbola
    """
    (A, B, C, D, E) = coeff
    solved = fsolve(_project_data_func, x, [x, y, coeff])
    hyperbola_x = solved[0]

    assert len(solved) == 1, f"More than one solution found for closest point to the hyperbola form {x,y}, {solved}, {coeff}"

    discrim = ((B*hyperbola_x+E)**2
            -4*C*(A*hyperbola_x**2
            +D*hyperbola_x-1))
    hyperbola_y = ((-B*hyperbola_x-E+np.sqrt(discrim))
                    /(2*C))
    
    distance = np.sqrt((x-hyperbola_x)**2
                        +(y-hyperbola_y)**2)

    if y >= hyperbola_y: # if the data point is on the inside of the jaw
        return (hyperbola_x, distance)
    else:
        return (hyperbola_x, -distance)

    

def _project_data_func(t: float, 
                       args: tuple[float, float, tuple[float, float, float, float, float]]
                       ) -> float:
    """
    Used for the coordinate t that gives the closest distance to point x y

    Params
    ------
    t: guess for closests point on the hyperbola to x, y
    args: [x, y, coeff] where
        1. x, y = data point to be projected
        2. coeff = coefficients of hyperbola

    Return
    ------
    dot product of the vector from x, y to the hyperbola at x = t and the tangent 
    vector of the hyperbola at x = t
    """
    (x, y, coeff) = args
    (A, B, C, D, E) = coeff
    discrim = ((B *t + E)**2
               -4*C * (A *t**2 + D *t -1))
    tangent = (1, (1/(2*C) * 
                   (-B + ((2*B * (B *t + E)) -4*C * (2*A *t + D))/(2 * np.sqrt(discrim)))))
    normal = (t - x, ((-B *t - E + np.sqrt(discrim))/(2*C)) - y)
    return tangent[0]*normal[0] + tangent[1]*normal[1]

def project_arclength(x: float, y: float, 
                      coeff: tuple[float, float, float, float, float]
                      ) -> tuple[np.array, np.array, float, tuple[float, float], tuple[float, float]]:
    """
    Compute the coordinates of a strip of 2*CONFIG.SAMPLING_WIDTH pixels in 
    the normal directions of the hyperbola. 

    Params
    ------
    x: x-coordinate of the point of projection 
    y: y-coordinate of the point of projection 
    coeff: coefficients of hyperbola (A, B, C, D, E) for hyperbola having the 
    equation Ax^2 + Bxy + Cy^2 + Dx + Ey - 1 = 0

    Returns
    -------
    1. x- and y-coordinates of the sampled strip
    2. orthonormal vector (outer?) at the point of projection (x, y)
    3. tangent vector at the point of projection (x, y)
    """
    (A, B, C, D, E) = coeff
    
    discrim = ((B *x + E)**2
               -4*C * (A *x**2 + D *x -1))
    tangent = (1, (1/(2*C) * 
                   (-B + ((2*B * (B *x + E)) -4*C * (2*A *x + D))/(2 * np.sqrt(discrim)))))
    normal = (1, -1/(tangent[1]))
    normal_h = (normal[0]/np.sqrt((normal[0])**2+(normal[1])**2), 
                normal[1]/np.sqrt((normal[0])**2+(normal[1])**2))

    normal_x = []
    normal_y = []

    c = CONFIG.SAMPLING_WIDTH
    while c >= -CONFIG.SAMPLING_WIDTH:
        normal_x.append(int(x+c*normal_h[0]))
        normal_y.append(int(y+c*normal_h[1]))
        c -= 1
    
    df = pd.DataFrame({'x': normal_x, 'y': normal_y})
    df.sort_values(by=['y'], inplace=True)

    return (df['x'].to_numpy(), df['y'].to_numpy(), normal_h, tangent)