- This program is written in Python 3.10.10

## Initialization

- Create virtual machine (VM)
  > python3 -m venv venv (also try replacing python3 with python or python3.10)
- Start VM
  > source venv/bin/activate (mac/linux)
  > venv\Scripts\activate.bat (windows)
- Install Dependencies
  > pip install -r requirements.txt
- You might get a "notice" saying you should update pip. If so, update pip then install dependencies again:
  > pip install --upgrade pip
- To initialize folder structure, simply run any command adhering to the Command Format (see "Recurring Start" section)
  > python main.py (most basic version)
- Below demonstrates the output folder structure. Note that you must have all the .py files for the program to run normally.

## Recurring Start

- Every new session, restart the virtual environment by running
  > source venv/bin/activate (mac/linux)
  > venv\Scripts\activate.bat (windows)
- This can be deactivated with
  > deactivate
- Put raw images in the "/img" folder. If one doesn't exist, make one or run
  > python main.py
- Command Format:
  > python main.py -n 1 -s 1 [keyword] [keyword]...
- "-n" and "-s" are flags representing the "number" of images to process and the "starting location" from which to process the images. The above example will process 1 image starting at the 2nd image file (first location is 0; the images are ordered lexigraphically, not by date! They sort similarly but are not exactly identical).
- The keywords determine what processes to run. If none is specified, all processes except "analyze" is ran.
- Keywords:
  - match, manual, fitproj, format, analyze

## Introducing the Processes

Match: performs "template matching" which finds matches of the templates in the template folder onto the images to automate data collection
Manual: manual intervention to edit the collected data
FitProj: processes collected data, identify central tooth, and converts collected data into processed data
Format: concatenates processed data across different days and visualizes results
Analyze: an interactive way for data exploration

## Using the "Manual" GUI

- left click: plot / unplot data
- tab: alternates between tooth mode, gap mode, center tooth mode, center gap mode, changing the behavior of left click
- 1, 2, 3, 4: also alternates between tooth mode, gap mode, center tooth mode, center gap mode
- space: enter / exit viewing mode where boxes are hidden to better identify teeth
- esc: exit without saving
- s: exit and save
- left arrow key: backtrack to previous image (lexigraphically) by first saving and closing the current image's GUI window
- right arrow key: open the next image (lexigraphically) by first saving and closing the current image's GUI window

## Using the "Analyze" GUI

- left click: open projected image of the data of the day of interest
- left click + drag: open projected images of the data of the days of interest (stacked)
- right click: open the manual GUI to edit data (changes not applied on plots until "format" or "analyze" is ran again)

## Notes:

- Analyze first runs format then opens the interactive tool; thus running format before analyze is redundant
- Control + C and close any outstanding GUI or matplotlib windows to force exit / restart the program
- Many variable settings can be found in utils.py
