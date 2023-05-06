import os 
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


def continuity_filter(csv):
    GRAD_THRESHOLD = 5
    GRAD_EVEN_THRESHOLD = 50
    SMOOTH_THRESHOLD = 0.5
    SMOOTH_EVEN_THRESHOLD = 5
    WIDTH_SIZE = 15
    HEIGHT_SIZE = 7
    df = pd.read_csv(csv)
    df['x'] = df['x']+df['w']/2
    df['y'] = df['y']+df['h']/2

    x = df['x']
    y = df['y']
    df['gradient'] = np.gradient(y, x)
    df['smoothness'] = np.gradient(df['gradient'], x)
    df['gradient_even'] = np.gradient(y)
    df['smoothness_even'] = np.gradient(df['gradient'])

    # plt.style.use('bmh')
    fig, ((ax1, ax2, ax3, ax4),(ax5, ax6, ax7, ax8), (ax9, ax10, ax11, ax12)) = plt.subplots(3, 4)
    fig.set_figwidth(WIDTH_SIZE)
    fig.set_figheight(HEIGHT_SIZE)
    fig.suptitle('Noise Filtering', fontweight='bold', fontname='Times New Roman')
    ax1.plot(x,y,'.-b')
    ax1.set_title("Original", fontsize=10, fontweight="bold", fontname='Times New Roman')

    ax2.plot(x,df['gradient'],'.-r', label='gradient')
    ax2.plot(x,df['gradient_even'],'.-g', label='gradient even')
    ax3.plot(x,df['smoothness'],'.-r', label='smoothness')
    ax3.plot(x,df['smoothness_even'],'.-g', label='smoothness even')
    ax2.legend(fontsize=10)
    ax3.legend(fontsize=10)
    ax2.set_title("Gradient Raw", fontsize=10, fontweight="bold", fontname='Times New Roman')
    ax3.set_title("Smoothness Raw", fontsize=10, fontweight="bold", fontname='Times New Roman')

    df_grad = df.copy()
    df_grad.drop(df_grad[df_grad['gradient'] > GRAD_THRESHOLD].index, inplace=True)
    df_grad.drop(df_grad[df_grad['gradient'] < -GRAD_THRESHOLD].index, inplace=True)
    df_grad = df_grad[df_grad['gradient'].notna()]
    x_grad = df_grad['x']
    y_grad = df_grad['y']
    gradient = df_grad['gradient']

    df_smooth = df.copy()
    df_smooth.drop(df_smooth[df_smooth['smoothness'] > SMOOTH_THRESHOLD].index, inplace=True)
    df_smooth.drop(df_smooth[df_smooth['smoothness'] < -SMOOTH_THRESHOLD].index, inplace=True)
    df_smooth = df_smooth[df_smooth['smoothness'].notna()]
    x_smooth = df_smooth['x']
    y_smooth = df_smooth['y']
    smoothness = df_smooth['smoothness']

    df_grad_even = df.copy()
    df_grad_even.drop(df_grad_even[df_grad_even['gradient_even'] > GRAD_EVEN_THRESHOLD].index, inplace=True)
    df_grad_even.drop(df_grad_even[df_grad_even['gradient_even'] < -GRAD_EVEN_THRESHOLD].index, inplace=True)
    df_grad_even = df_grad_even[df_grad_even['gradient_even'].notna()]
    x_grad_even = df_grad_even['x']
    y_grad_even = df_grad_even['y']
    gradient_even = df_grad_even['gradient_even']

    df_smooth_even = df.copy()
    df_smooth_even.drop(df_smooth_even[df_smooth_even['smoothness_even'] > SMOOTH_EVEN_THRESHOLD].index, inplace=True)
    df_smooth_even.drop(df_smooth_even[df_smooth_even['smoothness_even'] < -SMOOTH_EVEN_THRESHOLD].index, inplace=True)
    df_smooth_even = df_smooth_even[df_smooth_even['smoothness_even'].notna()]
    x_smooth_even = df_smooth_even['x']
    y_smooth_even = df_smooth_even['y']
    smoothness_even = df_smooth_even['smoothness_even']

    ax5.plot(x_grad,y_grad,'.-b')
    ax6.plot(x_grad,gradient,'.-r')
    ax5.set_title("Gradient Filtering", fontsize=10, fontweight="bold", fontname='Times New Roman')
    ax6.set_title(f"Filtered Gradient: Threshold {GRAD_THRESHOLD}", fontsize=10, fontweight="bold", fontname='Times New Roman')

    ax7.plot(x_grad_even,y_grad_even,'.-b')
    ax8.plot(x_grad_even,gradient_even,'.-g')
    ax7.set_title("Even Gradient Filtering", fontsize=10, fontweight="bold", fontname='Times New Roman')
    ax8.set_title(f"Filtered Even Gradient: Threshold {GRAD_EVEN_THRESHOLD}", fontsize=10, fontweight="bold", fontname='Times New Roman')
    
    ax9.plot(x_smooth, y_smooth, '.-b')
    ax10.plot(x_smooth, smoothness, '.-r')
    ax9.set_title("Smoothness Filtering", fontsize=10, fontweight="bold", fontname='Times New Roman')
    ax10.set_title(f"Filtered Smoothness: Threshold {SMOOTH_THRESHOLD}", fontsize=10, fontweight="bold", fontname='Times New Roman')

    ax11.plot(x_smooth_even,y_smooth_even,'.-b')
    ax12.plot(x_smooth_even,smoothness_even,'.-g')
    ax11.set_title("Even Smoothness Filtering", fontsize=10, fontweight="bold", fontname='Times New Roman')
    ax12.set_title(f"Filtered Even Smoothness: Threshold {SMOOTH_EVEN_THRESHOLD}", fontsize=10, fontweight="bold", fontname='Times New Roman')

    fig.tight_layout()
    plt.savefig(f"{csv[0:len(csv)-4]}_filtered.png")
    # plt.show()
    return (x_grad,y_grad)
        


def graph_filter():
    pass


if __name__ == "__main__":
    data = [file for file in os.listdir(os.getcwd()) if file[len(file)-4:] == ".csv"]
    for csv in data:
        continuity_filter(csv)