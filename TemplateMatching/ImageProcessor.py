# library imports
import os 
import cv2
import time
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import warnings

#helper functions
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

        for directory in {self._PATH_MATCHING, 
                          self._PATH_MANUAL, 
                          self._PATH_FILTER, 
                          self._PATH_FIT, 
                          self._PATH_PROJECTION}:
            make_dir(directory)


        assert os.path.isfile(os.path.join(self._PATH_ROOT,'img', self.file_name)), f"'{self.file_name}' does not exist in img"

        # image and image projection data 
        self.image = cv2.imread(os.path.join('img', file_name), cv2.IMREAD_GRAYSCALE)
        self.image_proj = None

        self.height = self.image.shape[0]
        self.width = self.image.shape[1]

    #-----------------------------------------------------------------


    def template_matching(self, display_time: bool = False, mode: Match = Match.TWO_D) -> None:
        """
        Performs template matching and stores output in /processed/template matching/[img_name]. 

        If mode = Match.ONE_D, matches "template 1D" images to projected image in 
        /processed/projection/[img_name]. 
        If Match.TWO_D, matches "template" images to original image in /img. 
        
        Params
        ------
        display_time: display time to run function
        mode: one of Match.TWO_D or Match_ONE_D

        Notes
        -----
        Useful Links:
        https://docs.opencv.org/4.x/d4/dc6/tutorial_py_templateMatching.html (a tutorial for template matching)
        https://docs.opencv.org/4.x/df/dfb/group__imgproc__object.html#gga3a7850640f1fe1f58fe91a2d7583695da5be00b45a4d99b5e42625b4400bfde65 (equations for each algorithm)
        """

        start_time = time.time()
        
        # load projected image if mode is Match.ONE_D
        projected_img_path = os.path.join(self._PATH_PROJECTION, f"projection.{self.file_type}")
        if (mode == Match.ONE_D) and (self.image_proj is None):
            if os.path.isfile(projected_img_path):
                    self.image_proj = cv2.imread(projected_img_path, cv2.IMREAD_GRAYSCALE)
            else:
                raise RuntimeError(f"projected image {projected_img_path} not found; have you ran fit project?")
            
        # set up config thresholds, image data, template path
        if mode == Match.TWO_D:
            threshold = CONFIG.THRESHOLD
            iou_threshold = CONFIG.IOU_THRESHOLD
            templates = [file for file in os.listdir(self._PATH_TEMPLATE) if suffix(file) in CONFIG.FILE_TYPES]
            img = self.image
            template_dir = self._PATH_TEMPLATE
        elif mode == Match.ONE_D:
            threshold = CONFIG.THRESHOLD_1D
            iou_threshold = CONFIG.IOU_THRESHOLD_1D
            templates = [file for file in os.listdir(self._PATH_TEMPLATE_1D) if suffix(file) in CONFIG.FILE_TYPES]
            img = self.image_proj
            template_dir = self._PATH_TEMPLATE_1D

        teeth = []
        for template_path in templates:

            # load template and template dimensions
            template = cv2.imread(os.path.join(template_dir, template_path),cv2.IMREAD_GRAYSCALE)
            template_h, template_w = template.shape

            for method in CONFIG.METHODS:
                img_clone = img.copy()
                matching_score = cv2.matchTemplate(img_clone, template, method)
                #returns locations where matching_score is bigger than THRESHOLD
                filtered_matches = np.where(matching_score >= threshold)

                # for each (x,y)
                for pt in zip(*filtered_matches[::-1]):
                    intersect = False
                    for tooth in teeth[::]:
                        if self._intersection_over_union([pt[0], pt[1], template_w, template_h,
                                                matching_score[pt[1]][pt[0]]], tooth) > iou_threshold:
                            # if a location that intersects has a better matching score, replace
                            if matching_score[pt[1]][pt[0]] > tooth[4]:
                                teeth.remove(tooth)
                            else: 
                                intersect = True
                    
                    # if no intersection, add to list of teeth
                    if not intersect:
                        new_tooth = [pt[0], pt[1], template_w, template_h, matching_score[pt[1]][pt[0]], template_path]
                        teeth.append(new_tooth)
            


        # for each identified tooth, draw a rectangle, store data point
        matched_image = img.copy()
        template_matching_data = {'x':[],'y':[], 'w':[],'h':[], 'score':[], 'match':[]}
        for pt in teeth:
            cv2.rectangle(matched_image, (pt[0], pt[1]), (pt[0] + pt[2], pt[1] + pt[3]), (255,255,0), 2)
            template_matching_data['x'].append(pt[0])
            template_matching_data['y'].append(pt[1])
            template_matching_data['w'].append(CONFIG.SQUARE)
            template_matching_data['h'].append(CONFIG.SQUARE)
            template_matching_data['score'].append(pt[4])
            template_matching_data['match'].append(pt[5])
        df = pd.DataFrame(data=template_matching_data)
        df.sort_values(by=['x'], inplace=True)

        # save data and matching image
        os.chdir(self._PATH_MATCHING)
        if mode == Match.TWO_D:
            df.to_csv("template matching.csv")
            cv2.imwrite(f"template matching{self.file_type}", matched_image)
        elif mode == Match.ONE_D:
            df.to_csv("template matching 1D.csv")
            cv2.imwrite(f"template matching 1D{self.file_type}", matched_image)
        os.chdir(self._PATH_ROOT)


        if display_time:
            print(f"MATCH       | '{self.file_name}: {time.time()-start_time} s")
        end_procedure()  


    #---------------------------------------------------------------------------------------------------

    def filter(self, display_time: bool = False) -> None:
        """
        """

        start_time = time.time()

        # check required raw data exists, read data
        if CONFIG.FILTER == Filter.MANUAL:
            raw_data_path = os.path.join(self._PATH_MANUAL, f"manual data.csv")
            assert os.path.isfile(raw_data_path), f"'manual data.csv' does not exist in /processed/manual/{self.img_name} - did you run manual first?"
            df = pd.read_csv(raw_data_path)
        else:
            raw_data_path = os.path.join('processed', "template matching", self.img_name,f"template matching.csv")
            assert os.path.isfile(raw_data_path), f"'template matching.csv' does not exist in /processed/template matching/{self.img_name} - did you run match first?"
            df = pd.read_csv(os.path.join(self._PATH_MATCHING, f"template matching.csv"))

        # stores "centered points"
        dfFilter = pd.DataFrame()
        dfFilter['x'] = df['x']+df['w']/2
        dfFilter['y'] = df['y']+df['h']/2

        x_raw = dfFilter['x']
        y_raw = dfFilter['y']

        dfFilter['gradient'] = np.gradient(y_raw, x_raw)
        dfFilter['smoothness'] = np.gradient(dfFilter['gradient'], x_raw)
        dfFilter['gradientEven'] = np.gradient(y_raw)
        dfFilter['smoothnessEven'] = np.gradient(dfFilter['gradient'])

        fig, ((ax1, ax2, ax3, ax4),(ax5, ax6, ax7, ax8), (ax9, ax10, ax11, ax12)) = plt.subplots(3, 4)
        fig.tight_layout()
        fig.set_figwidth(CONFIG.WIDTH_SIZE)
        fig.set_figheight(CONFIG.HEIGHT_SIZE)


        fig.suptitle('Noise Filtering', fontweight='bold', fontname='Times New Roman')
        ax1.plot(x_raw, y_raw,'.-b')
        ax1.set_title("Original", fontsize=10, fontweight="bold", fontname='Times New Roman')

        # plotting raw data
        ax2.plot(x_raw, dfFilter['gradient'],'.-r', label='gradient')
        ax2.plot(x_raw, dfFilter['gradientEven'],'.-g', label='gradient even')
        ax3.plot(x_raw, dfFilter['smoothness'],'.-r', label='smoothness')
        ax3.plot(x_raw, dfFilter['smoothnessEven'],'.-g', label='smoothness even')
        ax2.legend(fontsize=10)
        ax3.legend(fontsize=10)
        ax2.set_title("Gradient Raw", fontsize=10, fontweight="bold", fontname='Times New Roman')
        ax3.set_title("Smoothness Raw", fontsize=10, fontweight="bold", fontname='Times New Roman')

        # gradient filtering
        dfGrad = dfFilter.copy()
        dfGrad.drop(dfGrad[dfGrad['gradient'] > CONFIG.GRAD_THRESHOLD].index, inplace=True)
        dfGrad.drop(dfGrad[dfGrad['gradient'] < -CONFIG.GRAD_THRESHOLD].index, inplace=True)
        dfGrad = dfGrad[dfGrad['gradient'].notna()]
        xGrad = dfGrad['x']
        yGrad = dfGrad['y']
        gradient = dfGrad['gradient']

        # smootherness filtering
        dfSmooth = dfFilter.copy()
        dfSmooth.drop(dfSmooth[dfSmooth['smoothness'] > CONFIG.SMOOTH_THRESHOLD].index, inplace=True)
        dfSmooth.drop(dfSmooth[dfSmooth['smoothness'] < -CONFIG.SMOOTH_THRESHOLD].index, inplace=True)
        dfSmooth = dfSmooth[dfSmooth['smoothness'].notna()]
        xSmooth = dfSmooth['x']
        ySmooth = dfSmooth['y']
        smoothness = dfSmooth['smoothness']

        # gradient even filtering
        dfGradEven = dfFilter.copy()
        dfGradEven.drop(dfGradEven[dfGradEven['gradientEven'] > CONFIG.GRAD_EVEN_THRESHOLD].index, inplace=True)
        dfGradEven.drop(dfGradEven[dfGradEven['gradientEven'] < -CONFIG.GRAD_EVEN_THRESHOLD].index, inplace=True)
        dfGradEven = dfGradEven[dfGradEven['gradientEven'].notna()]
        xGradEven = dfGradEven['x']
        yGradEven = dfGradEven['y']
        gradientEven = dfGradEven['gradientEven']

        # smootherness even filtering
        dfSmoothEven = dfFilter.copy()
        dfSmoothEven.drop(dfSmoothEven[dfSmoothEven['smoothnessEven'] > CONFIG.SMOOTH_EVEN_THRESHOLD].index, inplace=True)
        dfSmoothEven.drop(dfSmoothEven[dfSmoothEven['smoothnessEven'] < -CONFIG.SMOOTH_EVEN_THRESHOLD].index, inplace=True)
        dfSmoothEven = dfSmoothEven[dfSmoothEven['smoothnessEven'].notna()]
        xSmoothEven = dfSmoothEven['x']
        ySmoothEven = dfSmoothEven['y']
        smoothnessEven = dfSmoothEven['smoothnessEven']

        # plotting filtered data
        ax5.plot(xGrad,yGrad,'.-b')
        ax6.plot(xGrad,gradient,'.-r')
        ax5.set_title("Gradient Filtering", fontsize=10, fontweight="bold", 
                      fontname='Times New Roman')
        ax6.set_title(f"Filtered Gradient: Threshold {CONFIG.GRAD_THRESHOLD}", 
                      fontsize=10, fontweight="bold", fontname='Times New Roman')

        ax7.plot(xGradEven,yGradEven,'.-b')
        ax8.plot(xGradEven,gradientEven,'.-g')
        ax7.set_title("Even Gradient Filtering", fontsize=10, fontweight="bold", 
                      fontname='Times New Roman')
        ax8.set_title(f"Filtered Even Gradient: Threshold {CONFIG.GRAD_EVEN_THRESHOLD}", 
                      fontsize=10, fontweight="bold", fontname='Times New Roman')
        
        ax9.plot(xSmooth, ySmooth, '.-b')
        ax10.plot(xSmooth, smoothness, '.-r')
        ax9.set_title("Smoothness Filtering", fontsize=10, fontweight="bold", 
                      fontname='Times New Roman')
        ax10.set_title(f"Filtered Smoothness: Threshold {CONFIG.SMOOTH_THRESHOLD}", 
                       fontsize=10, fontweight="bold", fontname='Times New Roman')

        ax11.plot(xSmoothEven,ySmoothEven,'.-b')
        ax12.plot(xSmoothEven,smoothnessEven,'.-g')
        ax11.set_title("Even Smoothness Filtering", fontsize=10, fontweight="bold", 
                       fontname='Times New Roman')
        ax12.set_title(f"Filtered Even Smoothness: Threshold {CONFIG.SMOOTH_EVEN_THRESHOLD}", 
                       fontsize=10, fontweight="bold", fontname='Times New Roman')

        # saving data
        os.chdir(self._PATH_FILTER)
        plt.savefig(f"analysis{self.file_type}")
        dfFilter.to_csv(f"raw.csv")
        dfGrad.to_csv(f"grad filtered.csv")
        dfGradEven.to_csv(f"grad even filtered.csv")
        dfSmooth.to_csv(f"smooth filtered.csv")
        dfSmoothEven.to_csv(f"smooth even filtered.csv")
        os.chdir(self._PATH_ROOT)

        if display_time:
            print(f"FILTER      | '{self.file_name}': {time.time()-start_time} s")
        end_procedure()

    #---------------------------------------------------------------------------------------------------



    def fitProject(self, display_time: bool = False):
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

        if display_time:
            print(f"FIT PROJECT | '{self.file_name}': {time.time()-start_time} s")
        end_procedure()


    def manual(self, display_time: bool = False, mode = Match.ONE_D):
        """
        runs the GUI for manual editing; 
        if Match.ONE_D, uses data from "manual 1D data" if exists, else uses data from "projection data"
        if Match.TWO_D, uses data from "manual data" if exists, else uses data from "match data"
        """
        start_time = time.time()

        try:
            GUI.GUI(self.file_name, self.img_name, self.file_type, mode)
        except RuntimeError as err:
            print(err)
        if display_time:
            print(f"MANUAL      | '{self.file_name}': {time.time()-start_time} s")
        end_procedure()


    @staticmethod
    def plotResult(display_time: bool = False):
        '''
        plot the result from "manual data 1D"
        '''
        start_time = time.time()
        try:
            plot_result.dataToCSV()
        except RuntimeError as err:
            print(err)
        if display_time:
            print(f"PLOT MANUAL | {time.time()-start_time} s")
        end_procedure()


    @staticmethod
    def _intersection_over_union(p1: list[int, int, int, int], p2: list[int, int, int, int]) -> float:
        """
        returns the intersection area over the union area of two template matches (intersection
        over union score)

        Params
        ------
        p1: [x, y, w, h] (box 1)
        p2: [x, y, w, h] (box 2)
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