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
import plot_result
from utils import Match, CONFIG, Filter, make_dir, suffix, end_procedure


class ImageProcessor:
    """
    An ImageProcessor contains all functionalies of processing a wax print. 

    It's assumed that images are in "img"

    Properties
    ----------
    root_dir: root directory of program 
    file_type: image file-extension 
    file_name: name of image file with file-extension
    img_name: name of image file without file-extension
    image: actual image data
    height: height of image (pixels)
    width: width of image (pixels)
    """

    _PATH_ROOT = os.getcwd()
    _PATH_IMG = os.path.join(_PATH_ROOT, "img")
    _PATH_TEMPLATE = os.path.join(_PATH_ROOT, "template")
    _PATH_TEMPLATE_1D = os.path.join(_PATH_ROOT, "template 1D")

    def __init__(self, file_name: str) -> None:
        """
        Constructor for Image Processor

        Params
        ------
        file_name: name of image file
        """
        self.file_type = os.path.splitext(file_name)[1]
        self.file_name = file_name
        self.img_name = file_name.replace(self.file_type, "")

        # PATHS
        self._PATH_MATCHING = os.path.join(self._PATH_ROOT, "processed", "template matching", self.img_name)
        self._PATH_MANUAL = os.path.join(self._PATH_ROOT, "processed", "manual", self.img_name)
        self._PATH_FILTER = os.path.join(self._PATH_ROOT, "processed", "filter", self.img_name)
        self._PATH_FIT = os.path.join(self._PATH_ROOT, "processed", "fit", self.img_name)
        self._PATH_PROJECTION = os.path.join(self._PATH_ROOT, "processed", "projection", self.img_name)
        self._PATH_OUTPUT = os.path.join(self._PATH_ROOT, "processed", "output", self.img_name)

        make_dir(self._PATH_MATCHING)
        make_dir(self._PATH_MANUAL)
        make_dir(self._PATH_FILTER)
        make_dir(self._PATH_FIT)
        make_dir(self._PATH_PROJECTION)
        make_dir(self._PATH_OUTPUT)


        assert os.path.isfile(os.path.join(self._PATH_ROOT,'img', self.file_name)), f"'{self.file_name}' does not exist in img"

        # image and image projection data 
        self.image = cv2.imread(os.path.join('img', file_name), cv2.IMREAD_GRAYSCALE)
        self.image_proj = None

        self.height = self.image.shape[0]
        self.width = self.image.shape[1]


    def match(self, displayTime: bool = False, mode = Match.TWO_D):
        """
        Performs template matching and stores output in /processed/template matching. If 
        mode = Match.ONE_D, matches "template 1D" images to projected image. If Match.TWO_D, 
        matches "template" images to original image. 
        """
        start_time = time.time()
        
        # if we want to match the projected image but it's data is not loaded yet
        if (mode == Match.ONE_D) and (self.image_proj is None):
            try:
                if os.path.isfile(os.path.join(self._PATH_PROJECTION, f"projection.{self.file_type}")):
                    self.image_proj = os.path.join(self._PATH_PROJECTION, f"projection.{self.file_type}")
                    template_matching.templateMatching(self, mode)
                else:
                    raise RuntimeError("projected image not found; have you ran fit project?")
            except RuntimeError as err:
                print(err)
        else:
            template_matching.templateMatching(self, mode)
        
        if displayTime and mode == Match.TWO_D:
            print(f"MATCH       | '{self.file_name}: {time.time()-start_time} s")
        if displayTime and mode == Match.ONE_D:
            print(f"MATCH 1D    | '{self.file_name}': {time.time()-start_time} s")

        end_procedure()  


    def filter(self, displayTime: bool = False):
        '''
        Filter current image's data according to CONFIG.FILTER and thresholds. If it's Filter.Manual or Filter.None
        don't filter. Output result in "filter data" 
        '''
        start_time = time.time()

        if CONFIG.FILTER == Filter.MANUAL:
            path = os.path.join('processed', "manual", self.img_name,f"manual data.csv")
            assert os.path.isfile(path), f"'manual data.csv' does not exist in /processed/manual/{self.img_name} - did you run manual first?"
        else:
            path = os.path.join('processed', "template matching", self.img_name,f"template matching.csv")
            assert os.path.isfile(path), f"'template matching.csv' does not exist in /processed/template matching/{self.img_name} - did you run match first?"
        noise_filtering.continuityFilter(self.img_name, self.file_type)
        if displayTime:
            print(f"FILTER      | '{self.file_name}': {time.time()-start_time} s")
        end_procedure()


    def fitProject(self, displayTime: bool = False):
        '''
        takes data from "filter data" and project. If CONFIG.FILTER == Filter.MANUAL, also project manual data. 
        '''

        start_time = time.time()
        path = os.path.join('processed', "filter", self.img_name, "raw.csv")
        assert os.path.isfile(path), f"filtered files do not exist in /processed/filter/{self.img_name} - did you run filter first?"
        try:
            hyperbola_solve.solve(self.file_name, self.img_name, self.file_type, self.height)
        except RuntimeError as err:
            print(err)

        if displayTime:
            print(f"FIT PROJECT | '{self.file_name}': {time.time()-start_time} s")
        end_procedure()


    def manual(self, displayTime: bool = False, mode = Match.ONE_D):
        '''
        runs the GUI for manual editing; 
        if Match.ONE_D, uses data from "manual 1D data" if exists, else uses data from "projection data"
        if Match.TWO_D, uses data from "manual data" if exists, else uses data from "match data"
        '''
        start_time = time.time()

        try:
            GUI.GUI(self.file_name, self.img_name, self.file_type, mode)
        except RuntimeError as err:
            print(err)
        if displayTime:
            print(f"MANUAL      | '{self.file_name}': {time.time()-start_time} s")
        end_procedure()


    @staticmethod
    def plotResult(displayTime: bool = False):
        '''
        plot the result from "manual data 1D"
        '''
        start_time = time.time()
        try:
            plot_result.dataToCSV()
        except RuntimeError as err:
            print(err)
        if displayTime:
            print(f"PLOT MANUAL | {time.time()-start_time} s")
        end_procedure()


# creates the folder structure
'''
--img (images to be processed)
--template (templates for matching on images in original form)
--template 1D (templates for matching on images in strip form)
--processed (folder where results are stored)
     -- match 
     -- filter 
     -- fit 
     -- projection 
     -- manual 
'''
current = os.getcwd()
make_dir("img")
make_dir("template")
make_dir("template 1D")
make_dir("processed")
os.chdir(os.path.join(current,"processed"))
make_dir("filter")
make_dir("fit")
make_dir("template matching")
make_dir("projection")
make_dir("manual")
make_dir("output")
os.chdir(current)

# suppresses warnings for a cleaner output (comment to unsuppress)
warnings.filterwarnings('ignore')
#set the theme for matplotlib plots
plt.style.use(CONFIG.PLOT_STYLE)