import os 
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from utils import CONFIG


def continuity_filter(file_name, img_name, manual):

    current_dir = os.getcwd()
    if manual:
        os.chdir(os.path.join(current_dir,'processed', "manual data"))
    else:
        os.chdir(os.path.join(current_dir,'processed', "match data"))
    df = pd.read_csv(f"{img_name}.csv")
    df_filter = pd.DataFrame()
    os.chdir(current_dir)

    df_filter['x'] = df['x']+df['w']/2
    df_filter['y'] = df['y']+df['h']/2

    x = df_filter['x']
    y = df_filter['y']
    df_filter['gradient'] = np.gradient(y, x)
    df_filter['smoothness'] = np.gradient(df_filter['gradient'], x)
    df_filter['gradient_even'] = np.gradient(y)
    df_filter['smoothness_even'] = np.gradient(df_filter['gradient'])

    fig, ((ax1, ax2, ax3, ax4),(ax5, ax6, ax7, ax8), (ax9, ax10, ax11, ax12)) = plt.subplots(3, 4)
    fig.set_figwidth(CONFIG.WIDTH_SIZE)
    fig.set_figheight(CONFIG.HEIGHT_SIZE)
    fig.suptitle('Noise Filtering', fontweight='bold', fontname='Times New Roman')
    ax1.plot(x,y,'.-b')
    ax1.set_title("Original", fontsize=10, fontweight="bold", fontname='Times New Roman')

    ax2.plot(x,df_filter['gradient'],'.-r', label='gradient')
    ax2.plot(x,df_filter['gradient_even'],'.-g', label='gradient even')
    ax3.plot(x,df_filter['smoothness'],'.-r', label='smoothness')
    ax3.plot(x,df_filter['smoothness_even'],'.-g', label='smoothness even')
    ax2.legend(fontsize=10)
    ax3.legend(fontsize=10)
    ax2.set_title("Gradient Raw", fontsize=10, fontweight="bold", fontname='Times New Roman')
    ax3.set_title("Smoothness Raw", fontsize=10, fontweight="bold", fontname='Times New Roman')

    df_grad = df_filter.copy()
    df_grad.drop(df_grad[df_grad['gradient'] > CONFIG.GRAD_THRESHOLD].index, inplace=True)
    df_grad.drop(df_grad[df_grad['gradient'] < -CONFIG.GRAD_THRESHOLD].index, inplace=True)
    df_grad = df_grad[df_grad['gradient'].notna()]
    x_grad = df_grad['x']
    y_grad = df_grad['y']
    gradient = df_grad['gradient']

    df_smooth = df_filter.copy()
    df_smooth.drop(df_smooth[df_smooth['smoothness'] > CONFIG.SMOOTH_THRESHOLD].index, inplace=True)
    df_smooth.drop(df_smooth[df_smooth['smoothness'] < -CONFIG.SMOOTH_THRESHOLD].index, inplace=True)
    df_smooth = df_smooth[df_smooth['smoothness'].notna()]
    x_smooth = df_smooth['x']
    y_smooth = df_smooth['y']
    smoothness = df_smooth['smoothness']

    df_grad_even = df_filter.copy()
    df_grad_even.drop(df_grad_even[df_grad_even['gradient_even'] > CONFIG.GRAD_EVEN_THRESHOLD].index, inplace=True)
    df_grad_even.drop(df_grad_even[df_grad_even['gradient_even'] < -CONFIG.GRAD_EVEN_THRESHOLD].index, inplace=True)
    df_grad_even = df_grad_even[df_grad_even['gradient_even'].notna()]
    x_grad_even = df_grad_even['x']
    y_grad_even = df_grad_even['y']
    gradient_even = df_grad_even['gradient_even']

    df_smooth_even = df_filter.copy()
    df_smooth_even.drop(df_smooth_even[df_smooth_even['smoothness_even'] > CONFIG.SMOOTH_EVEN_THRESHOLD].index, inplace=True)
    df_smooth_even.drop(df_smooth_even[df_smooth_even['smoothness_even'] < -CONFIG.SMOOTH_EVEN_THRESHOLD].index, inplace=True)
    df_smooth_even = df_smooth_even[df_smooth_even['smoothness_even'].notna()]
    x_smooth_even = df_smooth_even['x']
    y_smooth_even = df_smooth_even['y']
    smoothness_even = df_smooth_even['smoothness_even']

    ax5.plot(x_grad,y_grad,'.-b')
    ax6.plot(x_grad,gradient,'.-r')
    ax5.set_title("Gradient Filtering", fontsize=10, fontweight="bold", fontname='Times New Roman')
    ax6.set_title(f"Filtered Gradient: Threshold {CONFIG.GRAD_THRESHOLD}", fontsize=10, fontweight="bold", fontname='Times New Roman')

    ax7.plot(x_grad_even,y_grad_even,'.-b')
    ax8.plot(x_grad_even,gradient_even,'.-g')
    ax7.set_title("Even Gradient Filtering", fontsize=10, fontweight="bold", fontname='Times New Roman')
    ax8.set_title(f"Filtered Even Gradient: Threshold {CONFIG.GRAD_EVEN_THRESHOLD}", fontsize=10, fontweight="bold", fontname='Times New Roman')
    
    ax9.plot(x_smooth, y_smooth, '.-b')
    ax10.plot(x_smooth, smoothness, '.-r')
    ax9.set_title("Smoothness Filtering", fontsize=10, fontweight="bold", fontname='Times New Roman')
    ax10.set_title(f"Filtered Smoothness: Threshold {CONFIG.SMOOTH_THRESHOLD}", fontsize=10, fontweight="bold", fontname='Times New Roman')

    ax11.plot(x_smooth_even,y_smooth_even,'.-b')
    ax12.plot(x_smooth_even,smoothness_even,'.-g')
    ax11.set_title("Even Smoothness Filtering", fontsize=10, fontweight="bold", fontname='Times New Roman')
    ax12.set_title(f"Filtered Even Smoothness: Threshold {CONFIG.SMOOTH_EVEN_THRESHOLD}", fontsize=10, fontweight="bold", fontname='Times New Roman')

    fig.tight_layout()

    # saves to coordinates saves marked image in appropriate folders
    os.chdir(os.path.join(current_dir,'processed', "filter visualization"))
    plt.savefig(file_name)
    os.chdir(current_dir)

    os.chdir(os.path.join(current_dir,'processed', "filter data"))
    df_filter.to_csv(f"{img_name}.csv")
    df_grad.to_csv(f"{img_name}_grad.csv")
    df_grad_even.to_csv(f"{img_name}_gradeven.csv")
    df_smooth.to_csv(f"{img_name}_smooth.csv")
    df_smooth_even.to_csv(f"{img_name}_smootheven.csv")
    os.chdir(current_dir)        


def graph_filter():
    pass
