# library imports
import os 
import cv2
import time
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import warnings
from scipy.optimize import fsolve
from scipy.signal import find_peaks


#helper functions
from GUI import GUI
from utils import Match, CONFIG, Filter, Tooth, Cross
from helper import make_dir, suffix, end_procedure, \
    axis_symmetry, equidistant_set, intersection_over_union, \
    plot_hyperbola_linear, project_data_one, project_arclength


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
    image: image
    image_proj: projected image
    height: height of image (pixels)
    width: width of image (pixels)

    matching_data: template matching data
    manual_data: manual matching data
    filtered_data: filtered data
    manual_data_1D: manual matching data projected
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


        assert os.path.isfile(os.path.join(self._PATH_IMG, self.file_name)), f"'{self.file_name}' does not exist in img"

        # image and image projection data 
        image_proj_path = os.path.join(self._PATH_PROJECTION, f"projection{self.file_type}")
        self.image = cv2.imread(os.path.join(self._PATH_IMG, file_name))
        self.image_proj = cv2.imread(image_proj_path) if os.path.isfile(image_proj_path) else None
        self.height = self.image.shape[0]
        self.width = self.image.shape[1]

        # pandas data
        matching_data_path = os.path.join(self._PATH_MATCHING, "template matching.csv")
        manual_data_path = os.path.join(self._PATH_MANUAL, "manual data.csv")
        manual_data_path_1D = os.path.join(self._PATH_MANUAL, "manual data 1D.csv")
        filtered_data_path = os.path.join(self._PATH_FILTER, "raw.csv") # all files are created at the same time

        self.matching_data = pd.read_csv(matching_data_path) if os.path.isfile(matching_data_path) else None
        self.manual_data = pd.read_csv(manual_data_path) if os.path.isfile(manual_data_path) else None
        self.manual_data_1D = pd.read_csv(manual_data_path_1D) if os.path.isfile(manual_data_path_1D) else None
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
        # note that img is converted into grayscale for template matching
        
        if mode == Match.TWO_D:
            threshold = CONFIG.THRESHOLD
            iou_threshold = CONFIG.IOU_THRESHOLD
            templates = [file for file in os.listdir(self._PATH_TEMPLATE) if suffix(file) in CONFIG.FILE_TYPES]
            img = cv2.cvtColor(self.image.copy(), cv2.COLOR_BGR2GRAY)
            template_dir = self._PATH_TEMPLATE
        elif mode == Match.ONE_D:
            threshold = CONFIG.THRESHOLD_1D
            iou_threshold = CONFIG.IOU_THRESHOLD_1D
            templates = [file for file in os.listdir(self._PATH_TEMPLATE_1D) if suffix(file) in CONFIG.FILE_TYPES]
            img = cv2.cvtColor(self.image_proj.copy(), cv2.COLOR_BGR2GRAY)
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
                        if intersection_over_union([pt[0], pt[1], template_w, template_h,
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
            matching_data['x'].append(int(pt[0] + pt[2]/2))
            matching_data['y'].append(int(pt[1] + pt[3]/2))
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

        x_raw = df['x']
        y_raw = df['y']
        # stores "centered points"
        df_raw = pd.DataFrame()
        df_raw['x'] = x_raw 
        df_raw['y'] = y_raw 
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
            GUI(self.file_name, self.img_name, self.file_type)
        except RuntimeError as error:
            print(error)

        manual_data_path = os.path.join(self._PATH_MANUAL, "manual data.csv")
        self.manual_data = pd.read_csv(manual_data_path) if os.path.isfile(manual_data_path) else None

        if display_time:
            print(f"MANUAL      | '{self.file_name}': {time.time()-start_time} s")
        end_procedure()


    #---------------------------------------------------------------------------------------------------

    def fit_project(self, display_time: bool = False):
        """
        """
        start_time = time.time()

        # load the correct filtered data
        x = self.filtered_data["x"].to_numpy()
        y = self.filtered_data["y"].to_numpy()

        # matrix operations to solve for coefficient of best fitting conic; also solves for x-intercepts (bottom edge of image)
        matrixT = [x**2, x*y, y**2, x, y]
        matrix = np.transpose(matrixT)
        coeff = np.matmul(np.linalg.inv(np.matmul(matrixT, matrix)),np.matmul(matrixT, np.ones(np.shape(matrix)[0])))
        (A,B,C,D,E) = coeff
        x_inter = np.roots([A, self.height*B+D, -1+C*self.height**2+E*self.height])

        # if conic is a hyperbola, find equidistant set
        error = False
        if B**2-4*A*C < 0:
            hyperbola_fit = plot_hyperbola_linear(min(x_inter), max(x_inter), coeff)
            error = True
        else: 
            try: 
                hyperbola_fit = equidistant_set(min(x_inter), max(x_inter), coeff)
            except RuntimeError as err:
                raise RuntimeError(err)

        # plot fit
        img = self.image.copy()
        fig, ax = plt.subplots()
        ax.imshow(img, cmap=mpl.colormaps['gray'])
        ax.plot(hyperbola_fit[0], hyperbola_fit[1], '.-r', label="fit")
        
        # save figure 
        os.chdir(self._PATH_FIT)
        fig.savefig(f"fit{self.file_type}")
        os.chdir(self._PATH_ROOT)
        
        # if conic not a hyperbola, raise error
        if error:
            raise RuntimeError(f"""Unable to fit a Hyperbola or Parabola; Circle or Ellipse detected.
            See 'fit{self.file_type}' in /processed/fit/{self.img_name} for more detail.
            A={A}, B={B}, C={C}, D={D}, E={E}""")

        # pixel values of projected image
        proj_img = []
        # normal for every arclength point
        normal_xs = []
        normal_ys = []
        # tangent for every arclength point
        tangent_xs =[]
        tangent_ys = []

        for i in range(len(hyperbola_fit[0])):
            projection = project_arclength(hyperbola_fit[0][i], hyperbola_fit[1][i], coeff) # (x, y, normal, tagent)
            temp = []

            # append normal/tangent components
            normal_xs.append(projection[2][0])
            normal_ys.append(projection[2][1])
            tangent_xs.append(projection[3][0])
            tangent_ys.append(projection[3][1])

            for j in range(len(projection[0])):
                try:
                    # get the pixel value
                    pixel = img[projection[1][j],projection[0][j]]
                    temp.append([pixel[0], pixel[1], pixel[2]])
                except IndexError:
                    # if out of bounds, append WHITE
                    temp.append([255,255,255])
            
            proj_img.append(temp)
            # generates sampling plot
            ax.plot(projection[0], projection[1], '.-y', label="projection")

        # sets up a data frame to store projection data
        df_proj = pd.DataFrame()
        df_proj["arclength_loc"] = range(len(hyperbola_fit[0]))
        df_proj["x_2D"] = hyperbola_fit[0]
        df_proj["y_2D"] = hyperbola_fit[1]
        df_proj["tangent_x"] = tangent_xs
        df_proj["tangent_y"] = tangent_ys
        df_proj["normal_x"] = normal_xs
        df_proj["normal_y"] = normal_ys

        # save data and projection image
        os.chdir(self._PATH_PROJECTION)
        df_proj.to_csv(f"projection.csv")
        proj_img_transpose = cv2.transpose(np.array(proj_img))
        cv2.imwrite(f"projection{self.file_type}", proj_img_transpose)
        fig.savefig(f"projection sampling{self.file_type}")
        self.image_proj = cv2.imread(f"projection{self.file_type}")
        os.chdir(self._PATH_ROOT)


        if CONFIG.FILTER == Filter.MANUAL:
            
            teeth_df = pd.DataFrame()
            closest_proj_indecies = []
            closest_ys = []
            side = [CONFIG.SQUARE for _ in range(len(x))]

            for tooth_ind in range(len(x)):
                proj_data = project_data_one(x[tooth_ind], y[tooth_ind], coeff) #return (x, distance)
                closest_x = proj_data[0] 
                closest_y = proj_data[1]
                closest_proj_indecies.append(np.argmin([abs(i-closest_x) for i in hyperbola_fit[0]]))
                closest_ys.append(CONFIG.SAMPLING_WIDTH+closest_y)

            teeth_df["x"] = closest_proj_indecies
            teeth_df["y"] = closest_ys
            teeth_df["w"] = side
            teeth_df["h"] = side
            df_manual = self.manual_data
            center_ind = self.sum_dot_prod(coeff)

            t = df_manual.index[df_manual["type"]=="Tooth.CENTER_T"].to_numpy()
            g = df_manual.index[df_manual["type"]=="Tooth.CENTER_G"].to_numpy()
            assert len(t)+len(g) < 2, f"more than one center tooth or gap found in {self.img_name}.csv"

            # if center tooth doesn't exist 
            if len(t)+len(g) == 0:
                if df_manual["type"][center_ind] == "Tooth.TOOTH":
                    df_manual["type"][center_ind] = "Tooth.CENTER_T"
                elif df_manual["type"][center_ind] == "Tooth.GAP":
                    df_manual["type"][center_ind] = "Tooth.CENTER_G"
            else:
                if len(t) > 0 and center_ind != t[0] or len(g) > 0 and center_ind != g[0]:
                        print(f"Alternative center index found: {center_ind}; please ensure the current center index is correct")

            # find and mark potential errors
            potential_errors = self.arclength_histogram(closest_proj_indecies)
            for e in potential_errors:
                if df_manual["type"][e] == "Tooth.TOOTH":
                    df_manual["type"][e] = "Tooth.ERROR_T"
                elif df_manual["type"][e] == "Tooth.GAP":
                    df_manual["type"][e] = "Tooth.ERROR_G"

            teeth_df["type"] = df_manual["type"]

            self.manual_data = df_manual
            self.manual_data_1D = teeth_df
            # save and re-plot altered data
            self.update_data_plot(Match.TWO_D)
            self.update_data_plot(Match.ONE_D)
            
        if display_time:
            print(f"FIT/PROJECT | '{self.file_name}': {time.time()-start_time} s")
        end_procedure()


    def update_data_plot(self, mode: Match) -> None:
        """
        Plot df data onto image and save.

        If mode is Match.ONE_D, read 
            1) image from /processed/projection/[img_name]/projection[filetype];
        If mode is Match.TWO_D, read 
            1) image from /img/[file_name]
        
        Params
        ------
        mode: one of Match.ONE_D or Match.TWO_D
        """
        if mode == Match.ONE_D:
            if self.image_proj is None:
                raise RuntimeError(f"projected image for {self.img_name} is not found; did you run fit project first?")
            image = self.image_proj.copy()
            df = self.manual_data_1D
        else:
            image = self.image.copy()
            df = self.manual_data

        # reads dataframe
        x = df["x"].to_numpy()
        y = df["y"].to_numpy()
        w = df["w"].to_numpy()
        h = df["h"].to_numpy()
        type = np.array([])
        for i in df["type"]:
            if i == "Tooth.TOOTH":
                type = np.append(type, Tooth.TOOTH)
            elif i == "Tooth.GAP":
                type = np.append(type, Tooth.GAP)
            elif i == "Tooth.CENTER_T":
                type = np.append(type, Tooth.CENTER_T)
            elif i == "Tooth.CENTER_G":
                type = np.append(type, Tooth.CENTER_G)
            elif i == "Tooth.ERROR_T":
                type = np.append(type, Tooth.ERROR_T)
            elif i == "Tooth.ERROR_G":
                type = np.append(type, Tooth.ERROR_G)

        for i in range(len(x)):
                image = GUI.draw_tooth(image, int(x[i]), int(y[i]), w[i], h[i], type[i])

        GUI.save(self.file_name, self.img_name, self.file_type, mode, image, df)


    def arclength_histogram(self, arclength_location: list[float]) -> list[float]:
        """
        Create 4 distance histograms with 10, 20, 30, 40 bin respectively. Saves 
        results to '/processed/projection'. Returns indecies of teeth involved 
        in creating distances outside thresholds (potential errors). 

        Params
        ------
        arclength_location: a list arclength positions of the data points
        """
        
        distances = []
        for location in range(len(arclength_location)-1):
            distances.append(arclength_location[location+1] - arclength_location[location])


        hist, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, tight_layout=True)
        ax1.hist(distances, bins=10)
        ax2.hist(distances, bins=20)
        ax3.hist(distances, bins=30)
        ax4.hist(distances, bins=40)

        ax1.grid()
        ax2.grid()
        ax3.grid()
        ax4.grid()

        os.chdir(self._PATH_PROJECTION)
        hist.savefig(f"distance histogram{self.file_type}")
        os.chdir(self._PATH_ROOT)

        suspect_distances = [ind for ind in range(len(distances)) 
                            if (distances[ind] <= CONFIG.ERROR_LOWER_B or distances[ind] >= CONFIG.ERROR_UPPER_B)]

        res = []
        for dist in suspect_distances:
            res.append(dist)
            res.append(dist + 1)
        return res

        
    def sum_dot_prod(self, coeff: tuple[float, float, float, float, float]) -> int:
        """
        """
        axis_sym = axis_symmetry(coeff)
        x = self.filtered_data["x"].to_numpy()
        y = self.filtered_data["y"].to_numpy()

        points = CONFIG.CROSS * 2 + 1

        if len(x) < points: 
            raise RuntimeError(f"Unable to perform {CONFIG.CROSS} crosses with {len(x)} points; {points} points needed")
        
        center_index = CONFIG.CROSS
        dot_sums = []

        while center_index < len(x) - CONFIG.CROSS:
            dot_sum = 0
            for c in range(CONFIG.CROSS):
                p1i  = center_index + c + 1
                p2i =  center_index - c - 1

                vec1 = (x[p1i], y[p1i])
                vec2 = (x[p2i], y[p2i])

                vec_diff = np.subtract(vec1, vec2)            
                if CONFIG.CROSS_METHOD == Cross.SQAURED:
                    dot_sum += np.dot(vec_diff, axis_sym)**2
                elif CONFIG.CROSS_METHOD == Cross.ABS:
                    dot_sum += abs(np.dot(vec_diff, axis_sym))
            dot_sums.append(dot_sum)
            center_index += 1
        

        return np.argmin(dot_sums) + CONFIG.CROSS


    #---------------------------------------------------------------------------------------------------


    def avg_intensity(self) -> None:
        """
        """
        # don't really remember why but transpose here
        data = cv2.transpose(self.image_proj.copy())
        os.chdir(self._PATH_PROJECTION)
        fig2, ax2 = plt.subplots()

        fig2.set_figwidth(CONFIG.WIDTH_SIZE)
        fig2.set_figheight(CONFIG.HEIGHT_SIZE)
        fig2.tight_layout()

        avg_intensity = np.mean(data, axis=1)
        avg_window_intensity = []
        for i in range(len(avg_intensity)):
            if i - int(CONFIG.WINDOW_WIDTH/2) < 0:
                start = 0
            else: 
                start = i - int(CONFIG.WINDOW_WIDTH/2)
            if i + int(CONFIG.WINDOW_WIDTH/2) > len(data):
                end = len(data)
            else: 
                end = i + int(CONFIG.WINDOW_WIDTH/2)
            avg_window_intensity.append(np.mean(avg_intensity[start:end], axis=0))
        
        avg_intensity_graph = [255-i[0] for i in avg_window_intensity]

        # assumes image is at current dir
        if self.image_proj is None:
            raise RuntimeError(f"projected image for {self.img_name} is not found; did you run fit project first?")
        projection = self.image_proj.copy()

        ax2.imshow(projection)
        ax2.plot(range(len(avg_intensity_graph)),avg_intensity_graph, color='y')
        local_max_index, _ = find_peaks(avg_intensity_graph, distance=30)
        ax2.scatter(local_max_index,np.ones(len(local_max_index)), color='r')

        fig2.savefig(f"projection intensity{self.file_type}")
        os.chdir(self._PATH_ROOT)