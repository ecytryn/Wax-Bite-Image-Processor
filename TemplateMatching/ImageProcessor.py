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
from utils import Match, CONFIG, Filter, make_dir, suffix, end_procedure


# creates the folder structure
"""
--img (images to be processed)
--template (templates for matching on images in original form)
--template 1D (templates for matching on images in strip form)
--processed (folder where results are stored)
     -- match 
     -- filter 
     -- fit 
     -- projection 
     -- manual 
"""

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

        # pandas data
        matching_data_path = os.path.join(self._PATH_MATCHING, "template matching.csv")
        manual_data_path = os.path.join(self._PATH_MANUAL, "manual data.csv")
        filtered_data_path = os.path.join(self._PATH_FILTER, "raw.csv") # all files are created at the same time

        self.matching_data = pd.read_csv(matching_data_path) if os.path.isfile(matching_data_path) else None
        self.manual_data = pd.read_csv(manual_data_path) if os.path.isfile(manual_data_path) else None
        self.filtered_data = pd.read_csv(filtered_data_path) if os.path.isfile(filtered_data_path) else None


    #---------------------------------------------------------------------------------------------------


    def template_matching(self, display_time: bool = False, mode: Match = Match.TWO_D) -> None:
        """
        Performs template matching and stores output in '/processed/template matching/[img_name]' 

        If mode = Match.ONE_D, matches '/template 1D' images to projected image in 
        '/processed/projection/[img_name]'. If Match.TWO_D, matches '/template' images 
        to original image in '/img'. 
        
        Params
        ------
        display_time: display log of time to run function
        mode: one of Match.TWO_D or Match_ONE_D

        Notes
        -----
        Useful Links:
        https://docs.opencv.org/4.x/d4/dc6/tutorial_py_templateMatching.html (a tutorial for template matching)
        https://docs.opencv.org/4.x/df/dfb/group__imgproc__object.html#gga3a7850640f1fe1f58fe91a2d7583695da5be00b45a4d99b5e42625b4400bfde65 (equations for each algorithm)
        """

        start_time = time.time()
        
        if (mode == Match.ONE_D) and (self.image_proj is None):
            raise RuntimeError(f"projected image for {self.img_name} is not found; did you run fit project first?")
            
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
        matching_data = {'x':[],'y':[], 'w':[],'h':[], 'score':[], 'match':[]}
        for pt in teeth:
            cv2.rectangle(matched_image, (pt[0], pt[1]), (pt[0] + pt[2], pt[1] + pt[3]), (255,255,0), 2)
            matching_data['x'].append(int(pt[0] + pt[2]/2 - CONFIG.SQUARE/2))
            matching_data['y'].append(int(pt[1] + pt[3]/2 - CONFIG.SQUARE/2))
            matching_data['w'].append(CONFIG.SQUARE)
            matching_data['h'].append(CONFIG.SQUARE)
            matching_data['score'].append(pt[4])
            matching_data['match'].append(pt[5])
        df = pd.DataFrame(data=matching_data)
        df.sort_values(by=['x'], inplace=True)

        self.matching_data = df

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


    @staticmethod
    def _intersection_over_union(p1: list[int, int, int, int], p2: list[int, int, int, int]) -> float:
        """
        Computes the intersection area over the union area of two boxes ('intersection
        over union' score). Helper of template_matching. 

        Params
        ------
        p1: [x, y, w, h] (box 1)
        p2: [x, y, w, h] (box 2)

        Returns
        -------
        iou: intersection over union score
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


    #---------------------------------------------------------------------------------------------------


    def filter(self, display_time: bool = False) -> None:
        """
        Computes the gradient (first derivative), smoothness (second derivative), 
        gradient even, smoothness even (the last two assumes even spacing of teeth). 
        Then, filters these computations by the thresholds defined in CONFIG. Saves 
        data in '/processed/filter/{self.img_name}'.

        If CONFIG.FILTER is Filter.MANUAL, uses manually edited data in 
        '/processed/manual/{self.img_name}'. Otherwise, uses template matching data in 
        '/processed/template matching/{self.img_name}'. 

        Note: CONFIG is defined in utils.py

        Params
        ------
        display_time: display log of time to run function
        """
        start_time = time.time()

        # check required raw data exists
        if CONFIG.FILTER == Filter.MANUAL:
            raw_data_path = os.path.join(self._PATH_MANUAL, f"manual data.csv")
            assert os.path.isfile(raw_data_path), f"'manual data.csv' does not exist in /processed/manual/{self.img_name} - did you run manual first?"
            df = pd.read_csv(raw_data_path)
        else:
            if self.matching_data is None:
                raise f"template matching data does not exist for {self.img_name} - did you run match first?"
            df = self.matching_data

        # stores "centered points"
        df_raw = pd.DataFrame()
        df_raw['x'] = df['x']+df['w']/2
        df_raw['y'] = df['y']+df['h']/2

        x_raw = df_raw['x']
        y_raw = df_raw['y']

        df_raw['gradient'] = np.gradient(y_raw, x_raw)
        df_raw['smoothness'] = np.gradient(df_raw['gradient'], x_raw)
        df_raw['gradient_even'] = np.gradient(y_raw)
        df_raw['smoothness_even'] = np.gradient(df_raw['gradient'])

        # setting width, height, title and legend
        fig, ((ax1, ax2, ax3, ax4),(ax5, ax6, ax7, ax8), (ax9, ax10, ax11, ax12)) = plt.subplots(3, 4)
        fig.tight_layout()
        fig.set_figwidth(CONFIG.WIDTH_SIZE)
        fig.set_figheight(CONFIG.HEIGHT_SIZE)
        fig.suptitle('Noise Filtering', fontweight='bold', fontname='Times New Roman')
        ax1.set_title("Original", fontsize=10, fontweight="bold", fontname='Times New Roman')
        ax2.set_title("Gradient Raw", fontsize=10, fontweight="bold", fontname='Times New Roman')
        ax3.set_title("Smoothness Raw", fontsize=10, fontweight="bold", fontname='Times New Roman')
        ax5.set_title("Gradient Filtering", fontsize=10, fontweight="bold", 
                      fontname='Times New Roman')
        ax6.set_title(f"Filtered Gradient: Threshold {CONFIG.GRAD_THRESHOLD}", 
                      fontsize=10, fontweight="bold", fontname='Times New Roman')
        ax7.set_title("Even Gradient Filtering", fontsize=10, fontweight="bold", 
                      fontname='Times New Roman')
        ax8.set_title(f"Filtered Even Gradient: Threshold {CONFIG.GRAD_EVEN_THRESHOLD}", 
                      fontsize=10, fontweight="bold", fontname='Times New Roman')
        ax9.set_title("Smoothness Filtering", fontsize=10, fontweight="bold", 
                      fontname='Times New Roman')
        ax10.set_title(f"Filtered Smoothness: Threshold {CONFIG.SMOOTH_THRESHOLD}", 
                       fontsize=10, fontweight="bold", fontname='Times New Roman')
        ax11.set_title("Even Smoothness Filtering", fontsize=10, fontweight="bold", 
                       fontname='Times New Roman')
        ax12.set_title(f"Filtered Even Smoothness: Threshold {CONFIG.SMOOTH_EVEN_THRESHOLD}", 
                       fontsize=10, fontweight="bold", fontname='Times New Roman')
        
        # plotting raw data (unfiltered)
        ax1.plot(x_raw, y_raw,'.-b')
        ax2.plot(x_raw, df_raw['gradient'],'.-r', label='gradient')
        ax2.plot(x_raw, df_raw['gradient_even'],'.-g', label='gradient even')
        ax3.plot(x_raw, df_raw['smoothness'],'.-r', label='smoothness')
        ax3.plot(x_raw, df_raw['smoothness_even'],'.-g', label='smoothness even')

        ax2.legend(fontsize=10)
        ax3.legend(fontsize=10)

        # gradient filtering
        df_grad = self.filter_one(df_raw, CONFIG.GRAD_THRESHOLD, "gradient")
        x_grad = df_grad['x']
        y_grad = df_grad['y']
        gradient = df_grad['gradient']

        # smootherness filtering
        df_smooth = self.filter_one(df_raw, CONFIG.SMOOTH_THRESHOLD, "smoothness")
        x_smooth = df_smooth['x']
        y_smooth = df_smooth['y']
        smoothness = df_smooth['smoothness']

        # gradient even filtering
        df_grad_even = self.filter_one(df_raw, CONFIG.GRAD_EVEN_THRESHOLD, "gradient_even")
        x_grad_even = df_grad_even['x']
        y_grad_even = df_grad_even['y']
        gradient_even = df_grad_even['gradient_even']

        # smootherness even filtering
        df_smooth_even = self.filter_one(df_raw, CONFIG.SMOOTH_EVEN_THRESHOLD, "smoothness_even")
        x_smooth_even = df_smooth_even['x']
        y_smooth_even = df_smooth_even['y']
        smoothness_even = df_smooth_even['smoothness_even']

        if CONFIG.FILTER == Filter.GRADIENT:
            self.filtered_data = df_grad
        elif CONFIG.FILTER == Filter.GRADIENT_EVEN:
            self.filtered_data = df_grad_even
        elif CONFIG.FILTER == Filter.SMOOTH:
            self.filtered_data = df_smooth
        elif CONFIG.FILTER == Filter.SMOOTH_EVEN:
            self.filtered_data = df_smooth_even
        else:
            self.filtered_data = df_raw

        # plotting filtered data
        ax5.plot(x_grad, y_grad,'.-b')
        ax6.plot(x_grad, gradient,'.-r')
        ax7.plot(x_grad_even, y_grad_even,'.-b')
        ax8.plot(x_grad_even, gradient_even,'.-g')
        ax9.plot(x_smooth, y_smooth, '.-b')
        ax10.plot(x_smooth, smoothness, '.-r')
        ax11.plot(x_smooth_even, y_smooth_even,'.-b')
        ax12.plot(x_smooth_even, smoothness_even,'.-g')

        # saving data and analysis visualization
        os.chdir(self._PATH_FILTER)
        plt.savefig(f"analysis{self.file_type}")
        df_raw.to_csv(f"raw.csv")
        df_grad.to_csv(f"grad filtered.csv")
        df_grad_even.to_csv(f"grad even filtered.csv")
        df_smooth.to_csv(f"smooth filtered.csv")
        df_smooth_even.to_csv(f"smooth even filtered.csv")
        os.chdir(self._PATH_ROOT)

        if display_time:
            print(f"FILTER      | '{self.file_name}': {time.time()-start_time} s")
        end_procedure()


    @staticmethod
    def filter_one(df_raw: pd.DataFrame, threshold: float, key: str) -> pd.DataFrame:
        """
        Filtered a dataframe by a threshold and column. Helper for filter. 

        Params
        ------
        df_raw: dataframe to filter
        threshold: threshold of filter; rows with column values within 
        (-threshold, threshold) will be kept
        key: key indicating the column to perform filtering on (for example: "gradient")

        Returns
        -------
        df_filtered: filtered dataframe
        """

        df_filtered = df_raw.copy()
        df_filtered.drop(df_filtered[df_filtered[key] > threshold].index, inplace=True)
        df_filtered.drop(df_filtered[df_filtered[key] < -threshold].index, inplace=True)
        df_filtered = df_filtered[df_filtered[key].notna()]
        return df_filtered


    #---------------------------------------------------------------------------------------------------


    def manual(self, display_time: bool = False, mode: Match = Match.TWO_D):
        """
        Opens interface for manual editing (a GUI instance).
        """
        start_time = time.time()

        try:
            GUI.GUI(self.file_name, self.img_name, self.file_type, mode)
        except RuntimeError as error:
            print(error)

        manual_data_path = os.path.join(self._PATH_MANUAL, "manual data.csv")
        self.manual_data = pd.read_csv(manual_data_path) if os.path.isfile(manual_data_path) else None

        if display_time:
            print(f"MANUAL      | '{self.file_name}': {time.time()-start_time} s")
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
        except RuntimeError as error:
            print(error)

        if display_time:
            print(f"FIT PROJECT | '{self.file_name}': {time.time()-start_time} s")
        end_procedure()


    #---------------------------------------------------------------------------------------------------
