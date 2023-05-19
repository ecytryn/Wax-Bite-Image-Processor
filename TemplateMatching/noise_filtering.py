import os 
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from utils import CONFIG, Filter


def continuityFilter(imgName, fileType):
    '''
    
    '''
    
    currDir = os.getcwd()

    # read appropriate data 
    if CONFIG.FILTER == Filter.MANUAL:
        df = pd.read_csv(os.path.join(currDir,'processed', "manual", imgName, f"manual data.csv"))
    else:
        df = pd.read_csv(os.path.join(currDir,'processed', "template matching", imgName, f"template matching.csv"))

    dfFilter = pd.DataFrame()
    dfFilter['x'] = df['x']+df['w']/2
    dfFilter['y'] = df['y']+df['h']/2

    x = dfFilter['x']
    y = dfFilter['y']
    dfFilter['gradient'] = np.gradient(y, x)
    dfFilter['smoothness'] = np.gradient(dfFilter['gradient'], x)
    dfFilter['gradientEven'] = np.gradient(y)
    dfFilter['smoothnessEven'] = np.gradient(dfFilter['gradient'])

    fig, ((ax1, ax2, ax3, ax4),(ax5, ax6, ax7, ax8), (ax9, ax10, ax11, ax12)) = plt.subplots(3, 4)
    fig.set_figwidth(CONFIG.WIDTH_SIZE)
    fig.set_figheight(CONFIG.HEIGHT_SIZE)
    fig.suptitle('Noise Filtering', fontweight='bold', fontname='Times New Roman')
    ax1.plot(x,y,'.-b')
    ax1.set_title("Original", fontsize=10, fontweight="bold", fontname='Times New Roman')

    ax2.plot(x,dfFilter['gradient'],'.-r', label='gradient')
    ax2.plot(x,dfFilter['gradientEven'],'.-g', label='gradient even')
    ax3.plot(x,dfFilter['smoothness'],'.-r', label='smoothness')
    ax3.plot(x,dfFilter['smoothnessEven'],'.-g', label='smoothness even')
    ax2.legend(fontsize=10)
    ax3.legend(fontsize=10)
    ax2.set_title("Gradient Raw", fontsize=10, fontweight="bold", fontname='Times New Roman')
    ax3.set_title("Smoothness Raw", fontsize=10, fontweight="bold", fontname='Times New Roman')

    dfGrad = dfFilter.copy()
    dfGrad.drop(dfGrad[dfGrad['gradient'] > CONFIG.GRAD_THRESHOLD].index, inplace=True)
    dfGrad.drop(dfGrad[dfGrad['gradient'] < -CONFIG.GRAD_THRESHOLD].index, inplace=True)
    dfGrad = dfGrad[dfGrad['gradient'].notna()]
    xGrad = dfGrad['x']
    yGrad = dfGrad['y']
    gradient = dfGrad['gradient']

    dfSmooth = dfFilter.copy()
    dfSmooth.drop(dfSmooth[dfSmooth['smoothness'] > CONFIG.SMOOTH_THRESHOLD].index, inplace=True)
    dfSmooth.drop(dfSmooth[dfSmooth['smoothness'] < -CONFIG.SMOOTH_THRESHOLD].index, inplace=True)
    dfSmooth = dfSmooth[dfSmooth['smoothness'].notna()]
    xSmooth = dfSmooth['x']
    ySmooth = dfSmooth['y']
    smoothness = dfSmooth['smoothness']

    dfGradEven = dfFilter.copy()
    dfGradEven.drop(dfGradEven[dfGradEven['gradientEven'] > CONFIG.GRAD_EVEN_THRESHOLD].index, inplace=True)
    dfGradEven.drop(dfGradEven[dfGradEven['gradientEven'] < -CONFIG.GRAD_EVEN_THRESHOLD].index, inplace=True)
    dfGradEven = dfGradEven[dfGradEven['gradientEven'].notna()]
    xGradEven = dfGradEven['x']
    yGradEven = dfGradEven['y']
    gradientEven = dfGradEven['gradientEven']

    dfSmoothEven = dfFilter.copy()
    dfSmoothEven.drop(dfSmoothEven[dfSmoothEven['smoothnessEven'] > CONFIG.SMOOTH_EVEN_THRESHOLD].index, inplace=True)
    dfSmoothEven.drop(dfSmoothEven[dfSmoothEven['smoothnessEven'] < -CONFIG.SMOOTH_EVEN_THRESHOLD].index, inplace=True)
    dfSmoothEven = dfSmoothEven[dfSmoothEven['smoothnessEven'].notna()]
    xSmoothEven = dfSmoothEven['x']
    ySmoothEven = dfSmoothEven['y']
    smoothnessEven = dfSmoothEven['smoothnessEven']

    ax5.plot(xGrad,yGrad,'.-b')
    ax6.plot(xGrad,gradient,'.-r')
    ax5.set_title("Gradient Filtering", fontsize=10, fontweight="bold", fontname='Times New Roman')
    ax6.set_title(f"Filtered Gradient: Threshold {CONFIG.GRAD_THRESHOLD}", fontsize=10, fontweight="bold", fontname='Times New Roman')

    ax7.plot(xGradEven,yGradEven,'.-b')
    ax8.plot(xGradEven,gradientEven,'.-g')
    ax7.set_title("Even Gradient Filtering", fontsize=10, fontweight="bold", fontname='Times New Roman')
    ax8.set_title(f"Filtered Even Gradient: Threshold {CONFIG.GRAD_EVEN_THRESHOLD}", fontsize=10, fontweight="bold", fontname='Times New Roman')
    
    ax9.plot(xSmooth, ySmooth, '.-b')
    ax10.plot(xSmooth, smoothness, '.-r')
    ax9.set_title("Smoothness Filtering", fontsize=10, fontweight="bold", fontname='Times New Roman')
    ax10.set_title(f"Filtered Smoothness: Threshold {CONFIG.SMOOTH_THRESHOLD}", fontsize=10, fontweight="bold", fontname='Times New Roman')

    ax11.plot(xSmoothEven,ySmoothEven,'.-b')
    ax12.plot(xSmoothEven,smoothnessEven,'.-g')
    ax11.set_title("Even Smoothness Filtering", fontsize=10, fontweight="bold", fontname='Times New Roman')
    ax12.set_title(f"Filtered Even Smoothness: Threshold {CONFIG.SMOOTH_EVEN_THRESHOLD}", fontsize=10, fontweight="bold", fontname='Times New Roman')

    fig.tight_layout()

    # saving
    os.chdir(os.path.join(currDir,'processed', "filter", imgName))
    plt.savefig(f"analysis.{fileType}")
    dfFilter.to_csv(f"raw.csv")
    dfGrad.to_csv(f"grad filtered.csv")
    dfGradEven.to_csv(f"grad even filtered.csv")
    dfSmooth.to_csv(f"smooth filtered.csv")
    dfSmoothEven.to_csv(f"smooth even filtered.csv")
    os.chdir(currDir)

def graphFilter():
    pass
