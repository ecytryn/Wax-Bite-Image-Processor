# library imports 
import os

# modules
from ImageProcessor import ImageProcessor
from utils import Match, suffix, print_divider

if __name__ == "__main__":
    FILETYPE = [".jpg", ".jpeg", ".png"]
    images = [file for file in os.listdir(os.path.join(os.getcwd(),"img")) if suffix(file) in FILETYPE]
    print_divider()
    for image in images:
        process_img = ImageProcessor(image)
        # process_img.match(True, Match.TWO_D)
        # process_img.filter(True)
        # process_img.fit_project(True, 10)
        # process_img.match(True, Match.ONE_D)
        process_img.manual(True, Match.TWO_D)
        print_divider()