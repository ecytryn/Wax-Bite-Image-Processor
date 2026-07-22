import os
import time
from datetime import datetime, timedelta

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from GUI import GUI
from helper import parse_date, suffix
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from period_phase_asym import plot_period_phase_asym
from utils import CONFIG

DATA_DATES = []
ALL_DATES = []
FILE_NAMES = []


def format_result(display_time: bool = False) -> None:
    """
    Combined all output files into the desired format (one binary and one arclength result)
    """
    start_time = time.time()

    # finds path of all data files, sorted
    data_paths = search_file(CONFIG.RESULT_PATH, CONFIG.DATA_FILENAME)
    print(f"\n=== DEBUG: format_result ===")
    print(f"Found {len(data_paths)} data files")
    for i, path in enumerate(data_paths):
        print(f"  [{i}] {path}")
    
    max_center_index = 0
    max_length = 0
    center_indecies = []
    dates = []

    for i in range(len(data_paths)):
        # read file, parse date
        df = pd.read_csv(data_paths[i])
        subdirname = os.path.basename(os.path.dirname(data_paths[i]))
        date = parse_date(subdirname)
        dates.append(date)
        print(f"\n[Loop 1, i={i}] Processing '{subdirname}'")

        # find center index
        center_tooth = df.index[df["type"] == "Tooth.CENTER_T"].to_numpy()
        center_gap = df.index[df["type"] == "Tooth.CENTER_G"].to_numpy()
        center_nobite = df.index[df["type"] == "Tooth.CENTER_N"].to_numpy()
        
        print(f"  CENTER_T indices: {center_tooth}")
        print(f"  CENTER_G indices: {center_gap}")
        print(f"  CENTER_N indices: {center_nobite}")
        
        if len(center_tooth) > 0:
            center_index = center_tooth[0]
        elif len(center_gap) > 0:
            center_index = center_gap[0]
        elif len(center_nobite) > 0:
            center_index = center_nobite[0]
        else:
            print(f"  WARNING: No center marker found — SKIPPING")
            continue

        center_indecies.append(center_index)
        print(f"  Center index: {center_index}")

        # update max length, update max center index if applicable
        if center_index > max_center_index:
            max_center_index = center_index
        if len(df) > max_length:
            max_length = len(df)

    print(f"\n=== After Loop 1 ===")
    print(f"Total dates collected: {len(dates)}")
    print(f"Total center_indecies collected: {len(center_indecies)}")
    print(f"Dates: {[d.strftime('%Y_%m_%d') for d in dates]}")
    print(f"Center indices: {center_indecies}")
    
    # ... rest of function

    # setup dataframe to store desired data (1 column for date + the rest for teeth indecies)
    column_ids = np.array(range(max_center_index + max_length - 1)) - max_center_index
    columns_names = [str(column_id) for column_id in column_ids]
    df_output_arclength = pd.DataFrame(columns=["date"] + columns_names)
    df_output_binary = pd.DataFrame(columns=["date"] + columns_names)

    entry_so_far = 0
    for i in range(len(dates)):
        # read data
        df = pd.read_csv(data_paths[i])
        x = df["x"].to_numpy()
        types = df["type"]
        
        print(f"\n[Loop 2, i={i}] Date: {dates[i]}")
        print(f"  len(dates) = {len(dates)}")
        print(f"  len(center_indecies) = {len(center_indecies)}")
        print(f"  Accessing center_indecies[{i}] = {center_indecies[i]}")
        print(f"  df has {len(df)} rows")
        print(f"  len(x) = {len(x)}")

        # arclength representation = x values of projection (relative to center)
        arclength_data_rep = x - df["x"][center_indecies[i]]
        # binary data representation = 1 for teeth, 0 for gap
        binary_data_rep = []
        for i in range(len(x)):
            if types[i] in ("Tooth.TOOTH", "Tooth.CENTER_T", "Tooth.ERROR_T"):
                binary_data_rep.append(1)
            elif types[i] in ("Tooth.NO_BITE", "Tooth.CENTER_N"):
                binary_data_rep.append(float("nan"))
            else:
                binary_data_rep.append(0)

        # add front and back padding to obtain the correct shape to insert into dataframe
        arclength_data_rep_pad = padding(
            arclength_data_rep, center_indecies[i], max_center_index, len(columns_names)
        )
        binary_data_rep_pad = padding(
            binary_data_rep, center_indecies[i], max_center_index, len(columns_names)
        )

        # prepend date into the "date" column
        df_entry_arclength = [dates[i]] + arclength_data_rep_pad
        df_entry_binary = [dates[i]] + binary_data_rep_pad

        # add current file's entry into dataframe (at the last spot)
        df_output_arclength.loc[entry_so_far] = df_entry_arclength
        df_output_binary.loc[entry_so_far] = df_entry_binary
        entry_so_far += 1

    # save to output folder
    df_output_binary.sort_values(by=["date"], inplace=True)
    df_output_arclength.sort_values(by=["date"], inplace=True)

    # trim columns with na
    df_output_binary.dropna(axis=1, inplace=True)
    df_output_arclength.dropna(axis=1, inplace=True)

    df_output_binary.to_csv(os.path.join("processed", "output", "binary data.csv"))
    df_output_arclength.to_csv(
        os.path.join("processed", "output", "arclength data.csv")
    )

    if display_time:
        print(f"FORMAT      | {time.time() - start_time} s")


def format_erupfall(display_time: bool = False) -> None:
    """ """
    start_time = time.time()

    # checks if output data exist
    output_path = os.path.join("processed", "output")
    try:
        # df_arclength = pd.read_csv(os.path.join(output_path, "arclength data.csv"))
        df_binary = pd.read_csv(os.path.join(output_path, "binary data.csv"))
    except:
        raise RuntimeError(
            "Formatted output data do not exist in /processed/output. Did you run format_result?"
        )

    # parse string into dates from output file
    dates = [datetime.strptime(d, "%Y-%m-%d") for d in df_binary["date"]]
    # first two columns are: 1) index and 2) date, rest are teeth index
    index_columns = df_binary.columns.to_list()[2:]

    eruption_data = {"date": dates[1:]}
    fall_out_data = {"date": dates[1:]}

    for column_index in index_columns:
        # arclength_column = df_arclength[column_index].to_numpy()
        binary_column = df_binary[column_index].to_numpy()
        eruption_column = []
        fall_out_column = []

        no_erup_so_far = 0
        no_fall_so_far = 0
        last_known = binary_column[0]  # last non-NaN value

        for entry_index in range(1, len(dates)):
            curr_val = binary_column[entry_index]

            # skip NaN entries — increment counters but don't detect events
            if np.isnan(curr_val):
                no_erup_so_far += 1
                no_fall_so_far += 1
                eruption_column.append(0)
                fall_out_column.append(0)
                continue

            # if last_known is NaN (first entry was NaN), can't detect transitions
            if np.isnan(last_known):
                no_erup_so_far += 1
                no_fall_so_far += 1
                eruption_column.append(0)
                fall_out_column.append(0)
                last_known = curr_val
                continue

            if last_known == 0 and curr_val == 1:
                eruption_column.append(no_erup_so_far + 1)
                no_erup_so_far = 0
            else:
                no_erup_so_far += 1
                eruption_column.append(0)

            if last_known == 1 and curr_val == 0:
                fall_out_column.append(no_fall_so_far + 1)
                no_fall_so_far = 0
            else:
                no_fall_so_far += 1
                fall_out_column.append(0)

            last_known = curr_val

        eruption_data[column_index] = eruption_column
        fall_out_data[column_index] = fall_out_column

    # save to output folder
    df_eruption = pd.DataFrame(data=eruption_data)
    df_eruption.sort_values(by=["date"], inplace=True)
    df_eruption.to_csv(os.path.join("processed", "output", "eruption data.csv"))
    df_fall_out = pd.DataFrame(data=fall_out_data)
    df_fall_out.sort_values(by=["date"], inplace=True)
    df_fall_out.to_csv(os.path.join("processed", "output", "fall out data.csv"))

    if display_time:
        print(f"FORMAT ERUP | {time.time() - start_time} s")


def plot_result(display_time: bool = False) -> None:
    """
    Plot and save the formatted output data

    Params
    ------
    display_time: display log of time to run function
    """
    start_time = time.time()

    # checks if output data exist
    output_path = os.path.join("processed", "output")
    try:
        df_arclength = pd.read_csv(os.path.join(output_path, "arclength data.csv"))
        df_binary = pd.read_csv(os.path.join(output_path, "binary data.csv"))
        df_eruption = pd.read_csv(os.path.join(output_path, "eruption data.csv"))
        df_fall_out = pd.read_csv(os.path.join(output_path, "fall out data.csv"))
    except:
        raise RuntimeError(
            "Formatted output data do not exist in /processed/output. Did you run format_result?"
        )

    arc_tooth_x, arc_gap_x = [], []
    bin_tooth_x, bin_gap_x = [], []
    tooth_y, gap_y = [], []
    arc_nobite_x, bin_nobite_x, nobite_y = [], [], []

    # parse string into dates from output file

    dates = [datetime.strptime(d, "%Y-%m-%d") for d in df_arclength["date"]]
    # first two columns are: 1) index and 2) date, rest are teeth index
    index_columns = df_arclength.columns.to_list()[2:]

    for entry_index in range(len(dates)):
        for column_index in index_columns:
            # date = entry_index (the entry_indexth entry), tooth index = col_index
            # it's possible for it to not exist
            arc_entry = df_arclength[column_index][entry_index]
            bin_entry = df_binary[column_index][entry_index]

            # if exists
            if pd.isna(bin_entry):
                # no-bite marker
                arc_nobite_x.append(float(arc_entry))
                bin_nobite_x.append(float(column_index))
                nobite_y.append(dates[entry_index])
            else:
                # if a tooth
                if int(bin_entry) == 1:
                    arc_tooth_x.append(float(arc_entry))
                    bin_tooth_x.append(float(column_index))
                    tooth_y.append(dates[entry_index])
                # if a gap
                elif int(bin_entry) == 0:
                    arc_gap_x.append(float(arc_entry))
                    bin_gap_x.append(float(column_index))
                    gap_y.append(dates[entry_index])

    erup_x, nerup_x = [], []
    fall_x, nfall_x = [], []
    erup_y, nerup_y = [], []
    fall_y, nfall_y = [], []
    erupfall_dates = dates[1:]
    c_erupt = []
    c_fall = []

    for entry_index in range(len(erupfall_dates)):
        for column_index in index_columns:
            # date = entry_index (the entry_indexth entry), tooth index = col_index
            # it's possible for it to not exist
            erup_entry = df_eruption[column_index][entry_index]
            fall_entry = df_fall_out[column_index][entry_index]

            # if exists
            if not pd.isna(erup_entry):
                # if eruption
                if int(erup_entry) != 0:
                    erup_x.append(float(column_index))
                    erup_y.append(erupfall_dates[entry_index])
                    c_erupt.append(int(erup_entry))
                # if not eruption
                elif int(fall_entry) == 0:
                    nerup_x.append(float(column_index))
                    nerup_y.append(erupfall_dates[entry_index])

            # if exists
            if not pd.isna(fall_entry):
                # if eruption
                if int(fall_entry) != 0:
                    fall_x.append(float(column_index))
                    fall_y.append(erupfall_dates[entry_index])
                    c_fall.append(int(fall_entry))
                # if not eruption
                elif int(fall_entry) == 0:
                    nfall_x.append(float(column_index))
                    nfall_y.append(erupfall_dates[entry_index])

    # initialize figures
    ax_fig, arc_ax = plt.subplots()
    bin_fig, bin_ax = plt.subplots()
    erup_fig, erup_ax = plt.subplots()
    fall_fig, fall_ax = plt.subplots()

    # set dimensions and titles
    ax_fig.set_figwidth(CONFIG.WIDTH_SIZE)
    ax_fig.set_figheight(CONFIG.HEIGHT_SIZE)
    bin_fig.set_figwidth(CONFIG.WIDTH_SIZE)
    bin_fig.set_figheight(CONFIG.HEIGHT_SIZE)
    # to use the Cytrynbaum ratio
    # erup_fig.set_figwidth(15)
    # erup_fig.set_figheight(4)
    erup_fig.set_figwidth(CONFIG.WIDTH_SIZE)
    erup_fig.set_figheight(CONFIG.HEIGHT_SIZE)
    fall_fig.set_figwidth(CONFIG.WIDTH_SIZE)
    fall_fig.set_figheight(CONFIG.HEIGHT_SIZE)

    ax_fig.suptitle("Arclength Plot")
    bin_fig.suptitle("Index Plot")
    erup_fig.suptitle("Eruption Plot")
    fall_fig.suptitle("Fall Out Plot")

    # plot data
    arc_ax.scatter(arc_tooth_x, tooth_y, c="indigo", s=15)
    arc_ax.scatter(arc_gap_x, gap_y, c="lightgray", s=15)
    if nobite_y:
        arc_ax.scatter(arc_nobite_x, nobite_y, c="steelblue", s=15, marker="^")
    bin_ax.scatter(bin_tooth_x, tooth_y, c="indigo", s=15)
    bin_ax.scatter(bin_gap_x, gap_y, c="lightgray", s=15)
    if nobite_y:
        bin_ax.scatter(bin_nobite_x, nobite_y, c="steelblue", s=15, marker="^")
    erup_ax.scatter(erup_x, erup_y, c=c_erupt, s=50, cmap="winter")
    # erup_ax.scatter(nerup_x, nerup_y, c="lightgray", s=20)
    fall_ax.scatter(fall_x, fall_y, c=c_fall, s=50, cmap="winter")
    # fall_ax.scatter(nfall_x, nfall_y, c="lightgray", s=20)

    erup_fig.colorbar(
        ScalarMappable(norm=Normalize(min(c_erupt), max(c_erupt)), cmap="winter"),
        ax=erup_ax,
    )
    fall_fig.colorbar(
        ScalarMappable(norm=Normalize(min(c_fall), max(c_fall)), cmap="winter"),
        ax=fall_ax,
    )

    # set ticks from first to last date (for the grid)
    dates = sorted(dates)
    curr_date = dates[0]
    last_date = dates[-1]
    date_ticks = []
    while curr_date <= last_date:
        date_ticks.append(curr_date)
        curr_date += timedelta(days=3)

    # get minor and major grid lines
    teeth_index_ticks = np.linspace(-50, 50, num=101)
    teeth_arclength_ticks = np.linspace(-2500, 2500, num=101)

    arc_ax.set_xticks(teeth_arclength_ticks, minor=True)
    arc_ax.set_yticks(date_ticks, minor=True)
    bin_ax.set_xticks(teeth_index_ticks, minor=True)
    bin_ax.set_yticks(date_ticks, minor=True)
    erup_ax.set_xticks(teeth_index_ticks, minor=True)
    erup_ax.set_yticks(date_ticks, minor=True)
    fall_ax.set_xticks(teeth_index_ticks, minor=True)
    fall_ax.set_yticks(date_ticks, minor=True)

    # settings for minor and major grid lines
    arc_ax.grid(which="minor", color="k", linestyle=":", alpha=0.6)
    arc_ax.grid(which="major", color="k")
    bin_ax.grid(which="minor", color="k", linestyle=":", alpha=0.6)
    bin_ax.grid(which="major", color="k")
    # comment out the two lines below to get the Cytrybaum plot
    erup_ax.grid(which="minor", color="k", linestyle=":", alpha=0.6)
    erup_ax.grid(which="major", color="k")
    fall_ax.grid(which="minor", color="k", linestyle=":", alpha=0.6)
    fall_ax.grid(which="major", color="k")

    ax_fig.tight_layout()
    bin_fig.tight_layout()
    erup_fig.tight_layout()
    fall_fig.tight_layout()

    ax_fig.savefig(os.path.join(output_path, "arclength plot.png"))
    bin_fig.savefig(os.path.join(output_path, "index plot.png"))
    erup_fig.savefig(os.path.join(output_path, "eruption plot.png"))
    fall_fig.savefig(os.path.join(output_path, "fall out plot.png"))

    if display_time:
        print(f"PLOT RESULT | {time.time() - start_time} s")

    return (ax_fig, bin_fig, erup_fig, fall_fig)


# ---------------------------------------------------------------------


def analyze_result(display_time: bool = False) -> None:
    """
    Runs format_result and plot_result and open an interactive interface
    for quickly opening relevant visualizations. Note that opening the
    GUI and editing will not apply the effects immediately. Please run
    the "format" or "analyze" step again to update.

    left-click: opens projected image
    right-click: opens GUI image editor
    """
    global DATA_DATES, ALL_DATES, FILE_NAMES

    start_time = time.time()
    # set up
    format_result()
    format_erupfall()
    ax_fig, bin_fig, erup_fig, fall_fig = plot_result()

    # period / phase / asymmetry plot
    plot_period_phase_asym(
        os.path.join("processed", "output", "binary data.csv"),
        os.path.join("processed", "output"),
        include_mean=False,  # set to False to get just the 2-panel version
    )

    output_path = os.path.join("processed", "output")
    df_arclength = pd.read_csv(os.path.join(output_path, "arclength data.csv"))

    # update
    DATA_DATES = sorted(
        [datetime.strptime(d, "%Y-%m-%d") for d in df_arclength["date"]]
    )
    FILE_NAMES = [
        file
        for file in os.listdir(os.path.join(os.getcwd(), "img"))
        if suffix(file) in CONFIG.FILE_TYPES
    ]
    ALL_DATES = [parse_date(img_name) for img_name in FILE_NAMES]

    # connect event listners
    ax_fig.canvas.mpl_connect("button_press_event", _on_click)
    ax_fig.canvas.mpl_connect("button_release_event", _on_release)
    bin_fig.canvas.mpl_connect("button_press_event", _on_click)
    bin_fig.canvas.mpl_connect("button_release_event", _on_release)
    erup_fig.canvas.mpl_connect("button_press_event", _on_click)
    erup_fig.canvas.mpl_connect("button_release_event", _on_release)
    fall_fig.canvas.mpl_connect("button_press_event", _on_click)
    fall_fig.canvas.mpl_connect("button_release_event", _on_release)

    plt.show()
    if display_time:
        print(f"ANALYZE RES | {time.time() - start_time} s")


START_INDEX = None

CENTER_TYPES = ("Tooth.CENTER_T", "Tooth.CENTER_G", "Tooth.CENTER_N")


def _compute_indices(img_name: str) -> list[int] | None:
    """
    Read manual data 1D.csv for the given image and compute tooth indices
    relative to center (center = 0, left = negative, right = positive).

    Returns a list of indices aligned with the row order of the CSV,
    or None if the data file doesn't exist or has no center marker.
    """
    data_path = os.path.join("processed", "manual", img_name, "manual data 1D.csv")
    if not os.path.isfile(data_path):
        return None
    df = pd.read_csv(data_path)
    if "type" not in df.columns:
        return None

    # find center row
    center_row = None
    for ct in CENTER_TYPES:
        matches = df.index[df["type"] == ct].to_numpy()
        if len(matches) > 0:
            center_row = matches[0]
            break
    if center_row is None:
        return None

    return [i - center_row for i in range(len(df))]


def _label_1d_image(img_name: str, file_extension: str):
    """
    Read the 1D projected image and overlay tooth index labels.
    Returns the labeled image, or None if files are missing.
    """
    img_path = os.path.join(
        "processed", "manual", img_name, f"manual 1D{file_extension}"
    )
    if not os.path.isfile(img_path):
        return None

    img = cv2.imread(img_path)
    indices = _compute_indices(img_name)
    if indices is None:
        return img

    data_path = os.path.join("processed", "manual", img_name, "manual data 1D.csv")
    df = pd.read_csv(data_path)
    xs = df["x"].to_numpy()

    for i, idx in enumerate(indices):
        x_pos = int(xs[i])
        # draw index number above the strip center
        cv2.putText(
            img,
            str(idx),
            (x_pos - 5, 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
        )

    return img


def _build_labeled_stack(data_index: int, context: int = 2):
    """
    Build a vertically stacked image of labeled 1D strips for the given
    data_index ± context dates.

    Returns the stacked image and the window title string, or (None, None).
    """
    start = max(0, data_index - context)
    end = min(len(DATA_DATES) - 1, data_index + context)

    strips = []
    max_width = 0

    for idx in range(start, end + 1):
        date_index = ALL_DATES.index(DATA_DATES[idx])
        file_name = FILE_NAMES[date_index]
        img_name, file_ext = os.path.splitext(file_name)
        labeled = _label_1d_image(img_name, file_ext)
        if labeled is not None:
            if labeled.shape[1] > max_width:
                max_width = labeled.shape[1]
            strips.append(labeled)

    if not strips or max_width == 0:
        return None, None

    # resize all strips to same width
    resized_strips = []
    for s in strips:
        resized = cv2.resize(
            s, [max_width, CONFIG.SAMPLING_WIDTH * 2], interpolation=cv2.INTER_AREA
        )
        resized_strips.append(resized)

    stacked = np.concatenate(resized_strips, axis=0)
    title = DATA_DATES[data_index].strftime("%m_%d_%Y") + " (reference)"

    # apply MAX_WIDTH scaling
    if CONFIG.MAX_WIDTH is not None:
        ratio = CONFIG.MAX_WIDTH / stacked.shape[1]
        dimension = (CONFIG.MAX_WIDTH, int(stacked.shape[0] * ratio))
        stacked = cv2.resize(stacked, dimension, interpolation=cv2.INTER_AREA)

    return stacked, title


def _on_click(event) -> None:
    """
    Handler for button press event
    """
    global START_INDEX
    if event.ydata is not None:
        data_index, selected_date_index = _find_image_index(event.ydata)

        START_INDEX = data_index
        file_name = FILE_NAMES[selected_date_index]
        img_name, file_extension = os.path.splitext(file_name)

        if event.button == 3:
            # show labeled 1D reference alongside the 2D editor
            ref_img, ref_title = _build_labeled_stack(data_index)
            if ref_img is not None:
                cv2.imshow(ref_title, ref_img)
                cv2.moveWindow(ref_title, 50, 50)

            GUI(file_name, img_name, file_extension,
                FILE_NAMES, selected_date_index, False)

            # clean up reference window after GUI closes
            if ref_img is not None:
                cv2.destroyWindow(ref_title)


def _on_release(event) -> None:
    """
    Handler for button release event
    """
    global START_INDEX
    if event.ydata is not None and START_INDEX is not None:
        data_index, selected_date_index = _find_image_index(event.ydata)

        selected_indeces = []

        # this deals with drag from top to bottom or bottom to top
        if START_INDEX < data_index:
            start_index = START_INDEX
            end_index = data_index
        else:
            start_index = data_index
            end_index = START_INDEX

        START_INDEX = None

        # between the start and end files, also append everything in between
        for index in range(start_index, end_index + 1):
            selected_indeces.append(ALL_DATES.index(DATA_DATES[index]))

        if event.button == 1:
            max_width = 0

            # find max width (for resizing)
            for index in selected_indeces:
                curr_file_name = FILE_NAMES[index]
                curr_img_name, curr_file_extension = os.path.splitext(curr_file_name)
                curr_img = cv2.imread(
                    os.path.join(
                        os.getcwd(),
                        "processed",
                        "manual",
                        curr_img_name,
                        f"manual 1D{curr_file_extension}",
                    )
                )
                max_width = (
                    curr_img.shape[1] if curr_img.shape[1] > max_width else max_width
                )

            # set up first image
            img = []
            # stack images into img_resized vertically
            for curr_index in selected_indeces:
                img = _stack_img(img, max_width, curr_index)

            # resize window size
            if CONFIG.MAX_WIDTH is not None:
                ratio = CONFIG.MAX_WIDTH / img.shape[1]
                dimension = (CONFIG.MAX_WIDTH, int(img.shape[0] * ratio))
                resized = cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)
                cv2.imshow(DATA_DATES[start_index].strftime("%m_%d_%Y"), resized)
            else:
                cv2.imshow(DATA_DATES[start_index].strftime("%m_%d_%Y"), img)

            cv2.waitKey(0)
            cv2.destroyAllWindows()


def _find_image_index(ydata: float) -> int:
    """
    Find the index of the closest image in time to ydata (where the user clicked).

    Returns
    -------
    data_index: closest image index in the array of files with data
    selected_date_index: closest image index in img
    """
    days_delta = timedelta(days=int(ydata))
    start_time = datetime(year=1970, month=1, day=1)
    clicked_time = start_time + days_delta

    time_differences = np.absolute(np.array(DATA_DATES) - clicked_time)
    data_index = time_differences.argmin()
    selected_date = DATA_DATES[data_index]
    selected_date_index = ALL_DATES.index(selected_date)

    return (data_index, selected_date_index)


def _stack_img(img, max_width: int, curr_index: int):
    """
    Read projected image associated with curr_index. Resize to max_width. Stack new
    image on top of img. Overlays tooth index labels.

    Returns
    -------
    stacked image
    """
    curr_file_name = FILE_NAMES[curr_index]
    curr_img_name = os.path.splitext(curr_file_name)[0]
    curr_file_extension = os.path.splitext(curr_file_name)[1]

    labeled = _label_1d_image(curr_img_name, curr_file_extension)
    if labeled is None:
        labeled = cv2.imread(
            os.path.join(
                os.getcwd(),
                "processed",
                "manual",
                curr_img_name,
                f"manual 1D{curr_file_extension}",
            )
        )

    curr_img_resized = cv2.resize(
        labeled, [max_width, CONFIG.SAMPLING_WIDTH * 2], interpolation=cv2.INTER_AREA
    )
    if img != []:
        curr_img_resized = np.concatenate((curr_img_resized, img), axis=0)

    return curr_img_resized


# ---------------------------------------------------------------------
"HELPERS"
# ---------------------------------------------------------------------


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


def padding(
    unpadded_list: list, curr_center_ind: int, target_center_ind: int, num_of_cols: int
) -> list:
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
