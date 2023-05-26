import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import time
import cv2

from utils import CONFIG, Match
from helper import parse_date, suffix
from GUI import GUI


DATA_DATES = []
ALL_DATES = []
FILE_NAMES = []



def format_result() -> None:
    """
    Format output files into the desired format. 
    """

    # finds path of all data files, sorted
    data_paths = search_file(CONFIG.PATH, CONFIG.DATA_FILENAME)
    max_center_index = 0
    max_length = 0
    center_indecies = []
    dates = []

    for i in range(len(data_paths)):
        # read file
        df = pd.read_csv(data_paths[i])
        subdirname = os.path.basename(os.path.dirname(data_paths[i]))
        date = parse_date(subdirname)
        dates.append(date)

        # find center index
        center_tooth = df.index[df["type"] == "Tooth.CENTER_T"].to_numpy()
        if len(center_tooth) == 0:
            center_index = df.index[df["type"] == "Tooth.CENTER_G"].to_numpy()[0]
        else: 
            center_index = center_tooth[0]
        center_indecies.append(center_index)
        
        # update max length, update max center index
        if center_index > max_center_index:
            max_center_index = center_index
        if len(df) > max_length:
            max_length = len(df)

    # set dataframe shape (1 column for date + the rest for teeth indecies)
    columns = range(max_center_index + max_length - 1) - max_center_index
    columns = [str(column) for column in columns]
    df_output_arclength = pd.DataFrame(columns=["date"]+columns)
    df_output_binary = pd.DataFrame(columns=["date"]+columns)

    entry_so_far = 0
    for i in range(len(dates)):
        # read data
        df = pd.read_csv(data_paths[i])
        x = df["x"].to_numpy()
        types = df["type"]
        
        #arclength representation = x values of projection; binary data representation = 1 for teeth, 0 for gap
        arclength_data_rep = x - df["x"][center_indecies[i]]
        binary_data_rep = [1 if (types[i] == "Tooth.TOOTH" or types[i] == "Tooth.CENTER_T"
                                or types[i] == "Tooth.ERROR_T") else 0 for i in range(len(x))]

        # add front and back padding to obtain the correct shape to insert into dataframe
        arclength_data_rep_pad = padding(arclength_data_rep, center_indecies[i], max_center_index, len(columns))
        binary_data_rep_pad = padding(binary_data_rep, center_indecies[i], max_center_index, len(columns))

        # prepend date into the "date" column
        df_entry_arclength = [dates[i]] + arclength_data_rep_pad
        df_entry_binary = [dates[i]] + binary_data_rep_pad

        # set the maxlength'th entry to be the new entry; aka add new entry
        df_output_arclength.loc[entry_so_far] = df_entry_arclength
        df_output_binary.loc[entry_so_far] = df_entry_binary
        entry_so_far += 1

    # save to output folder
    df_output_binary.to_csv(os.path.join("processed", "output", "binary data.csv"))
    df_output_arclength.to_csv(os.path.join("processed", "output", "arclength data.csv"))



def plot_result() -> None:
    """
    Plot the formatted output data. 

    Requires
    --------
    format_result be ran first
    """

    output_path = os.path.join("processed", "output")
    df_arclength = pd.read_csv(os.path.join(output_path, "arclength data.csv"))
    dfBinary = pd.read_csv(os.path.join(output_path, "binary data.csv"))

    arcTooth = []
    arcGap = []
    arcCenterT = []
    arcCenterG = []

    binTooth = []
    binGap = []
    binCenterT = []
    binCenterG = []

    toothY = []
    gapY = []
    centerTY = []
    centerGY = []

    # convert string into dates
    dates = sorted([datetime.strptime(d, '%Y-%m-%d') for d in df_arclength["date"]])
    # first two columns are: 1) index of df, 2) date
    indexColumns = df_arclength.columns.to_list()[2:] 

    for entryIndex in range(len(dates)):
        for columnIndex in indexColumns:
            arcEntry = df_arclength[columnIndex][entryIndex]
            binEntry = dfBinary[columnIndex][entryIndex]

            # if not empty
            if not pd.isna(binEntry):
                # if a tooth
                if int(binEntry) == 1:
                    # if column is the centerindex
                    if int(columnIndex) == 0:
                        arcCenterT.append(float(arcEntry))
                        binCenterT.append(float(columnIndex))
                        centerTY.append(dates[entryIndex]) 
                    # if not a centertooth
                    else:
                        arcTooth.append(float(arcEntry))
                        binTooth.append(float(columnIndex))
                        toothY.append(dates[entryIndex])
                # if a gap
                elif int(binEntry) == 0:
                    # if column is centerindex
                    if int(columnIndex) == 0:
                        arcCenterG.append(float(arcEntry))
                        binCenterG.append(float(columnIndex))
                        centerGY.append(dates[entryIndex])
                    # if not a centergap
                    else:
                        arcGap.append(float(arcEntry))
                        binGap.append(float(columnIndex))
                        gapY.append(dates[entryIndex])

    ax_fig, arc_ax  = plt.subplots()
    bin_fig, bin_ax = plt.subplots()

    ax_fig.set_figwidth(CONFIG.WIDTH_SIZE)
    ax_fig.set_figheight(CONFIG.HEIGHT_SIZE)
    bin_fig.set_figwidth(CONFIG.WIDTH_SIZE)
    bin_fig.set_figheight(CONFIG.HEIGHT_SIZE)

    ax_fig.suptitle("Arclength Plot")
    bin_fig.suptitle("Index Plot")

    arc_ax.scatter(arcTooth, toothY, c="c", s=10)
    arc_ax.scatter(arcGap, gapY, c="#5A5A5A", s=10)
    arc_ax.scatter(arcCenterG, centerGY, c="#FFCCCB", s=10)
    arc_ax.scatter(arcCenterT, centerTY, c="r", s=10)

    bin_ax.scatter(binTooth, toothY, c="c", s=10)
    bin_ax.scatter(binGap, gapY, c="#5A5A5A", s=10)
    bin_ax.scatter(binCenterG, centerGY, c="#FFCCCB", s=10)
    bin_ax.scatter(binCenterT, centerTY, c="r", s=10)


    curr_date = dates[0]
    last_date = dates[-1]
    date_ticks = []
    while curr_date <= last_date:
        date_ticks.append(curr_date)
        curr_date += timedelta(days=3)

    teeth_index_ticks = np.linspace(-50, 50, num=101)
    teeth_arclength_ticks = np.linspace(-2500, 2500, num=101)

    arc_ax.set_xticks(teeth_arclength_ticks, minor=True)
    bin_ax.set_xticks(teeth_index_ticks, minor=True)
    arc_ax.set_yticks(date_ticks, minor=True)
    bin_ax.set_yticks(date_ticks, minor=True)

    arc_ax.grid(which='minor', color="k", linestyle=":", alpha=0.5)
    bin_ax.grid(which='minor', color="k", linestyle=":", alpha=0.5)
    arc_ax.grid(which='major', color="k", alpha=0.7)
    bin_ax.grid(which='major', color="k", alpha=0.7)


    ax_fig.tight_layout()
    bin_fig.tight_layout()

    ax_fig.savefig(os.path.join(output_path,"arclength plot.png"))
    bin_fig.savefig(os.path.join(output_path,"index plot.png"))

    return(ax_fig, arc_ax, bin_fig, bin_ax)


def analyze_result() -> None:
    global DATA_DATES, ALL_DATES, FILE_NAMES
    # set up 
    format_result()
    ax_fig, arc_ax, bin_fig, bin_ax = plot_result()

    output_path = os.path.join("processed", "output")
    df_arclength = pd.read_csv(os.path.join(output_path, "arclength data.csv"))
    DATA_DATES = sorted([datetime.strptime(d, '%Y-%m-%d') for d in df_arclength["date"]])

    FILE_NAMES = [file for file in os.listdir(os.path.join(os.getcwd(),"img")) 
                     if suffix(file) in CONFIG.FILE_TYPES]
    ALL_DATES = [parse_date(img_name) for img_name in FILE_NAMES]

    ax_fig.canvas.mpl_connect("button_press_event", _plot_button_handler)
    bin_fig.canvas.mpl_connect("button_press_event", _plot_button_handler)
    plt.show()


def _plot_button_handler(event) -> None:
    """
    """
    if event.ydata is not None:
        days_delta = timedelta(days = int(event.ydata))
        start_time = datetime(year=1970, month=1, day=1)
        clicked_time = start_time + days_delta

        time_differences = np.absolute(np.array(DATA_DATES)- clicked_time)
        selected_date = DATA_DATES[time_differences.argmin()]
        selected_date_index = ALL_DATES.index(selected_date)

        file_name = FILE_NAMES[selected_date_index]
        img_name = os.path.splitext(file_name)[0]
        file_extension = os.path.splitext(file_name)[1]

        if event.button == 1:
            img = cv2.imread(os.path.join(os.getcwd(), "processed", "manual", img_name, "manual 1D.jpg"))
            cv2.imshow(img_name, img)
            cv2.waitKey(0)
            cv2.destroyWindow(img_name)
        elif event.button == 3:
            GUI(file_name, img_name, file_extension)



#---------------------------------------
"HELPERS"
#----------------------------------------

def search_file(root: str, file_name: str) -> list[str]:
    """
    Finds all instances of a file within the root folder.

    Params
    ------
    root: path of root directory of search from
    file_name: file name 

    Returns
    -------
    A list of the full path of each instance of file_name.
    """

    # returns all directories and subdirectories in root
    all_dir = [x[0] for x in os.walk(root)]
    file_instances = []
    
    # iterates through all directories
    for dir in all_dir:
        items = os.listdir(dir)
        for item in items:
            # if file name is what we're looking for, append full path
            if item == file_name:
                file_instances.append(os.path.join(dir, item))
    
    return sorted(file_instances)



def padding(unpadded_list: list, curr_center_ind: int, target_center_ind: int, num_of_cols: int) -> list:
    """
    Pad unpadded_list so the center index is aligned with the target center
    index. 

    Params
    ------
    unpadded_list: list to be padded
    curr_center_ind: current center index
    target_center_ind: target center index
    num_of_cols: number of columns of the panda dataframe to insert the result
    list; the length needed for the padded list 

    Returns
    -------
    unpadded_list with padding such that the center index is aligned with the
    target index and the length is num_of_cols

    Note: curr_center_ind is not necessarily the "center" of the list.
    """

    unpadded_list = list(unpadded_list)

    front_pad_size = target_center_ind - curr_center_ind
    front_padding = [None for _ in range(front_pad_size)]

    back_pad_size = num_of_cols - len(front_padding) - len(unpadded_list)
    back_padding = [None for _ in range(back_pad_size)]
    
    return front_padding + unpadded_list + back_padding