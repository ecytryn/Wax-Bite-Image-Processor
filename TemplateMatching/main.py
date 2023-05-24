# library imports 
import os
import sys

# modules
from ImageProcessor import ImageProcessor
from utils import Match, CONFIG, suffix



def workflowOne(images):
    # remember to change FILTER to Filter.MANUAL in CONFIG
    match(images)
    manual(images)
    fitproj(images)
    format()

def match(images):
    for image in images:
        processImg = ImageProcessor(image)
        processImg.template_matching(True, Match.TWO_D)

def manual(images):
    for image in images:
        processImg = ImageProcessor(image)
        processImg.manual(True, Match.TWO_D)

def fitproj(images):
    for image in images:
        processImg = ImageProcessor(image)
        processImg.filter(True)
        processImg.fitProject(True)

def format():
    ImageProcessor.plotResult(True)


def flag_to_integer(args: list[str], flag: str):
    """
    """
    try:
        value = args[args.index(flag)+1]
        index = int(value)
    except ValueError:
        raise ValueError(f"{value} is not an integer.")
    except IndexError:
        raise IndexError(f"No integer followed by '-n'")

    if index < 0:
        raise RuntimeError(f"Flag {flag} is non positive")
    return index


if __name__ == "__main__":
    args = sys.argv

    # determine what to run
    match_bool = False
    manual_bool = False
    fitproj_bool = False
    format_bool = False
    if "match" in args:
        match_bool = True
    if "manual" in args:
        manual_bool = True
    if "fitproj" in args:
        fitproj_bool = True
    if "format" in args:
        format_bool = True
    

    args = [arg for arg in args if (arg not in {"match", "manual", "fitproj", "format", "main.py"})]
    images = sorted([file for file in os.listdir(os.path.join(os.getcwd(),"img")) if suffix(file) in CONFIG.FILE_TYPES])
    num_of_images = len(images)


    if "-s" in args or "-n" in args:
        start_index = 0
        num_to_process = num_of_images - start_index

        if "-s" in args:
            start_index = flag_to_integer(args, "-s")
        if "-n" in args:
            num_to_process = flag_to_integer(args, "-n")

        images = images[start_index : min(start_index+num_to_process,num_of_images)]

        if len(images) == 0:
            raise RuntimeError(f"Empty list detected: either starting index {start_index} is out of bounds (min = 0, max = {num_of_images-1}) or flag -n < 1")
        else:
            print(f"Processing images indexed {start_index} to {min(start_index+num_to_process-1,num_of_images-1)} (index starts at 0): '{images[0]}' to '{images[-1]}'")
    elif len(args) > 1:
        images = sorted([arg for arg in args if os.path.isfile(os.path.join("img", arg))])
        print(f"Processing images {images}")
    else:
        print(f"Processing all images")

    if not any([match_bool, manual_bool, fitproj_bool, format_bool]):
        print("Running: all")
        workflowOne(images)
    else:
        if match_bool:
            print("Running: Template Matching")
            match(images)
        if manual_bool:
            print("Running: Manual Processing")
            manual(images)
        if fitproj_bool:
            print("Running: Fit Project")
            fitproj(images)
        if format_bool:
            print("Running: Format")
            format()





