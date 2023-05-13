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
from utils import Match, PARAMS, make_dir, suffix, end_procedure


# creates the folder structure
# --img (images to be processed)
# --template (templates for matching on images in original form)
# --template 1D (templates for matching on images in strip form)
# --processed
#      -- match visualization
#      -- match data
#      -- match visualization
#      -- match data
#      -- match visualization
#      -- match data
#      -- match visualization
#      -- match data
#      -- match visualization
#      -- match data
#      -- match visualization
#      -- match data
current = os.getcwd()
make_dir("img")
make_dir("template")
make_dir("template 1D")
make_dir("processed")
os.chdir(os.path.join(current,"processed"))
for dir in PARAMS.DIRS_TO_MAKE:
    make_dir(dir)
os.chdir(current)

# suppresses warnings for a cleaner output (comment to unsuppress)
warnings.filterwarnings('ignore')
#set the theme for matplotlib plots
plt.style.use(PARAMS.PLOT_STYLE)


class ImageProcessor:
    
    def __init__(self, img_name: str):
        
        self.file_type = suffix(img_name)
        self.file_name = img_name
        self.name = img_name.replace(self.file_type, "")

        assert os.path.isfile(os.path.join(os.getcwd(),'img', self.file_name)), f"'{self.file_name}' does not exist"
        
        self.image = cv2.imread(os.path.join('img', img_name), cv2.IMREAD_GRAYSCALE)
        self.width = self.image.shape[1]
        self.height = self.image.shape[0]

    def match(self, displayTime: bool = False, mode = Match.TWO_D):
        start_time = time.time()
        if mode == Match.TWO_D:
            templates = [file for file in os.listdir(os.path.join(os.getcwd(),"template")) if file[len(file)-4:] == self.file_type]
            template_matching.template_matching(self.file_name, mode, templates, 0.75, 0.05)
        if mode == Match.ONE_D:
            templates = [file for file in os.listdir(os.path.join(os.getcwd(),"template 1D")) if file[len(file)-4:] == self.file_type]
            try:
                template_matching.template_matching(self.file_name, mode, templates, 0.75, 0.05)
            except RuntimeError as err:
                print(err)
        if displayTime and mode == Match.TWO_D:
            print(f"MATCH       | '{self.file_name}: {time.time()-start_time} s")
        if displayTime and mode == Match.ONE_D:
            print(f"MATCH 1D    | '{self.file_name}': {time.time()-start_time} s")
        end_procedure()  

    def filter(self,displayTime: bool = False, 
               gradthreshold: float = 5, gradeventhreshold: float = 50,
               smooththreshold: float = 0.5, smootheventhreshold: float = 5):
        start_time = time.time()
        path = os.path.join('processed', "match data",f"{self.name}.csv")
        assert os.path.isfile(path), f"'{self.name}.csv' does not exist - did you run match first?"
        noise_filtering.continuity_filter(self.file_name, self.name, gradthreshold, gradeventhreshold, 
                                          smooththreshold, smootheventhreshold)
        if displayTime:
            print(f"FILTER      | '{self.file_name}': {time.time()-start_time} s")
        end_procedure()

    def fit_project(self, displayTime: bool = False, window_width: int = 0):
        start_time = time.time()
        path = os.path.join('processed', "filter data",f"{self.name}.csv")
        img_path = os.path.join('img', self.file_name)
        assert os.path.isfile(path), f"'{self.name}.csv' does not exist - did you run filter first?"
        assert os.path.isfile(img_path), f"'{self.file_name}' does not exist - did you run filter first?"

        try:
            hyperbola_solve.solve(self.file_name, self.name, self.height, "grad", window_width)
        except RuntimeError as err:
            print(err)

        if displayTime:
            print(f"FIT PROJECT | '{self.file_name}': {time.time()-start_time} s")
        end_procedure()
    
    def manual(self, displayTime: bool = False, mode = Match.ONE_D):
        start_time = time.time()
        try:
            GUI.GUI(self.file_name, self.name, mode)
        except RuntimeError as err:
            print(err)
        if displayTime:
            print(f"MANUAL      | '{self.file_name}': {time.time()-start_time} s")
        end_procedure()
