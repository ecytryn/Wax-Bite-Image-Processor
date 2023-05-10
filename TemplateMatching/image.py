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
from dataclass import Match


class ImageProcessor:
    
    def __init__(self, img_name: str, file_type: str = None, plot_style: str = None):
        
        if file_type: 
            self.file_type = file_type
        else: 
            self.file_type = '.jpg'
        self.name = img_name[:len(img_name)-4]

        assert os.path.isfile(os.path.join(os.getcwd(),'img', img_name)), f"'{img_name}' does not exist"
        
        self.image = cv2.imread(os.path.join('img', img_name), cv2.IMREAD_GRAYSCALE)
        self.width = self.image.shape[1]
        self.height = self.image.shape[0]

        #set the theme for matplotlib plots
        if plot_style:
            plt.style.use(plot_style)
        else: 
            plt.style.use('bmh')

        current = os.getcwd()
        self.make_dir("img")
        self.make_dir("template")
        self.make_dir("template 1D")
        self.make_dir("processed")
        os.chdir(os.path.join(current,"processed"))
        self.make_dir("match visualization")
        self.make_dir("match data")
        self.make_dir("match visualization 1D")
        self.make_dir("match data 1D")
        self.make_dir("filter visualization")
        self.make_dir("filter data")
        self.make_dir("fit visualization")
        self.make_dir("projection")
        self.make_dir("projection sampling")
        self.make_dir("projection graphed")
        os.chdir(current)

    @staticmethod
    def make_dir(name: str):
        if not os.path.isdir(name):
            os.mkdir(name)

    def match(self, displayTime: bool = False, mode = Match.TWO_D):
        start_time = time.time()
        if mode == Match.TWO_D:
            templates = [file for file in os.listdir(os.path.join(os.getcwd(),"template")) if file[len(file)-4:] == self.file_type]
            template_matching.template_matching(f"{self.name}{self.file_type}", mode, templates, 0.75, 0.05)
        if mode == Match.ONE_D:
            templates = [file for file in os.listdir(os.path.join(os.getcwd(),"template 1D")) if file[len(file)-4:] == self.file_type]
            try:
                template_matching.template_matching(f"{self.name}{self.file_type}", mode, templates, 0.75, 0.05)
            except RuntimeError as err:
                print(err)
        if displayTime and mode == Match.TWO_D:
            print(f"MATCH       | '{self.name}{self.file_type}': {time.time()-start_time} s")
        if displayTime and mode == Match.ONE_D:
            print(f"MATCH 1D    | '{self.name}{self.file_type}': {time.time()-start_time} s")
        plt.close("all")
    

    def filter(self,displayTime: bool = False, 
               gradthreshold: float = 5, gradeventhreshold: float = 50,
               smooththreshold: float = 0.5, smootheventhreshold: float = 5):
        start_time = time.time()
        path = os.path.join('processed', "match data",f"{self.name}.csv")
        assert os.path.isfile(path), f"'{self.name}.csv' does not exist - did you run match first?"
        noise_filtering.continuity_filter(f"{self.name}.csv", gradthreshold, gradeventhreshold, 
                                          smooththreshold, smootheventhreshold)
        if displayTime:
            print(f"FILTER      | '{self.name}{self.file_type}': {time.time()-start_time} s")
        plt.close("all")


    def fit_project(self, displayTime: bool = False, intensity_window_width: int = 0):
        start_time = time.time()
        path = os.path.join('processed', "filter data",f"{self.name}.csv")
        img_path = os.path.join('img', f"{self.name}{self.file_type}")
        assert os.path.isfile(path), f"'{self.name}.csv' does not exist - did you run filter first?"
        assert os.path.isfile(img_path), f"'{self.name}{self.file_type}' does not exist - did you run filter first?"

        try:
            hyperbola_solve.solve(f"{self.name}.csv", self.file_type, self.height, "grad", intensity_window_width)
        except RuntimeError as err:
            print(err)

        if displayTime:
            print(f"FIT PROJECT | '{self.name}{self.file_type}': {time.time()-start_time} s")
        plt.close("all")


# suppresses warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    FILETYPE = ".jpg"
    images = [file for file in os.listdir(os.path.join(os.getcwd(),"img")) if file[len(file)-len(FILETYPE):] == FILETYPE]
    print("============================================================")
    for image in images:
        process_img = ImageProcessor(image, FILETYPE)
        # process_img.match(True, Match.TWO_D)
        # process_img.filter(True)
        # process_img.fit_project(True, 10)
        process_img.match(True, Match.ONE_D)
        print("============================================================")