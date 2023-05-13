# library imports
import os 
import cv2
import time
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import warnings

#helper functions
import template_matching
import noise_filtering
import hyperbola_solve
import GUI
from utils import Match, CONFIG, make_dir, suffix, end_procedure


# creates the folder structure
# --img (images to be processed)
# --template (templates for matching on images in original form)
# --template 1D (templates for matching on images in strip form)
# --processed (folder where results are stored)
#      -- match visualization (template matching result overlapped on original image)
#      -- match data (result data of template matching)
#      -- match visualization 1D (1D template matching result overlapped on projected image)
#      -- match data 1D (result data of 1D template matching)
#      -- filter visualization (visualization of the 4 filtering techniques)
#      -- filter data (result data after filtering)
#      -- fit visualization (fit conic overlapped on original image)
#      -- projection (projection result visualization)
#      -- projection sampling (visualization of the sampling done by projection)
#      -- projection graphed (intensity analysis visualization of projection)
#      -- projection data (data of projected points along conic and orthogonal vectors)
#      -- manual visualization (manual editing result overlapped on original image)
#      -- manual data (data of manual editing result)
#      -- manual visualization 1D (1D manual editing result overlapped on 1D image)
#      -- manual data 1D (data of 1D manual editing result)

current = os.getcwd()
make_dir("img")
make_dir("template")
make_dir("template 1D")
make_dir("processed")
os.chdir(os.path.join(current,"processed"))
for dir in CONFIG.DIRS_TO_MAKE:
    make_dir(dir)
os.chdir(current)

# suppresses warnings for a cleaner output (comment to unsuppress)
warnings.filterwarnings('ignore')
#set the theme for matplotlib plots
plt.style.use(CONFIG.PLOT_STYLE)


class ImageProcessor:
    
    def __init__(self, img_name: str):
        
        self.file_type = suffix(img_name)
        self.file_name = img_name
        self.img_name = img_name.replace(self.file_type, "")

        assert os.path.isfile(os.path.join(os.getcwd(),'img', self.file_name)), f"'{self.file_name}' does not exist"
        
        self.image = cv2.imread(os.path.join('img', img_name), cv2.IMREAD_GRAYSCALE)
        self.width = self.image.shape[1]
        self.height = self.image.shape[0]



    def match(self, display_time: bool = False, mode = Match.TWO_D):
        start_time = time.time()
        if mode == Match.TWO_D:
            templates = [file for file in os.listdir(os.path.join(os.getcwd(),"template")) if suffix(file) in CONFIG.FILE_TYPES]
        elif mode == Match.ONE_D:
            templates = [file for file in os.listdir(os.path.join(os.getcwd(),"template 1D")) if suffix(file) in CONFIG.FILE_TYPES]
        try:
            template_matching.template_matching(self.file_name, self.img_name, mode, templates)
        except RuntimeError as err:
            print(err)
        if display_time and mode == Match.TWO_D:
            print(f"MATCH       | '{self.file_name}: {time.time()-start_time} s")
        if display_time and mode == Match.ONE_D:
            print(f"MATCH 1D    | '{self.file_name}': {time.time()-start_time} s")
        end_procedure()  



    def filter(self, display_time: bool = False, manual: bool = False):
        start_time = time.time()
        if manual:
            path = os.path.join('processed', "manual data",f"{self.img_name}.csv")
            assert os.path.isfile(path), f"'{self.img_name}.csv' does not exist - did you run manual first?"
        else:
            path = os.path.join('processed', "match data",f"{self.img_name}.csv")
            assert os.path.isfile(path), f"'{self.img_name}.csv' does not exist - did you run match first?"
        noise_filtering.continuity_filter(self.file_name, self.img_name, manual)
        if display_time:
            print(f"FILTER      | '{self.file_name}': {time.time()-start_time} s")
        end_procedure()



    def fit_project(self, display_time: bool = False):
        start_time = time.time()
        path = os.path.join('processed', "filter data", f"{self.img_name}.csv")
        img_path = os.path.join('img', self.file_name)
        assert os.path.isfile(path), f"'{self.img_name}.csv' does not exist - did you run filter first?"
        assert os.path.isfile(img_path), f"'{self.file_name}' does not exist - did you run filter first?"
        try:
            hyperbola_solve.solve(self.file_name, self.img_name, self.height)
        except RuntimeError as err:
            print(err)

        if display_time:
            print(f"FIT PROJECT | '{self.file_name}': {time.time()-start_time} s")
        end_procedure()


    
    def manual(self, display_time: bool = False, mode = Match.ONE_D):
        start_time = time.time()
        try:
            GUI.GUI(self.file_name, self.img_name, mode)
        except RuntimeError as err:
            print(err)
        if display_time:
            print(f"MANUAL      | '{self.file_name}': {time.time()-start_time} s")
        end_procedure()
