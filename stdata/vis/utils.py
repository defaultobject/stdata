import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def tight_subplots(axes, fig, fig_width_pt=None, fig_height_pt=None, axis='x', _buffer=2, debug=False):
    """
    Automatically removes white space around a plot by ajusting the subplots to match the tightest box.
    Keeps the figure dimensions the same.
    """

    #figure out widht, height automatically so that it does not depend on backend
    if fig_width_pt is None:
        size = fig.get_size_inches()
        fig_width_pt = size[0]*fig.dpi
        fig_height_pt = size[1]*fig.dpi


    if type(axes) is not np.ndarray:
        #when using a single axis plot
        axes = np.array([[axes]])

    axes = axes.flatten()

    #find min, max values
    x0, x1, y0, y1 = [], [], [], []
    for ax in axes:
        bbox = ax.get_tightbbox(
            fig.canvas.get_renderer(),       
            call_axes_locator = True,  
            bbox_extra_artists = None
        )

        ax_x0 = bbox.xmin
        ax_y0 = bbox.ymin
        ax_x1= bbox.xmax
        ax_y1= bbox.ymax

        x0.append(ax_x0)
        x1.append(ax_x1)
        y0.append(ax_y0)
        y1.append(ax_y1)

    x0 = np.min(x0)
    x1 = np.max(x1)

    y0 = np.min(y0)
    y1 = np.max(y1)

    if debug:
        print('x0: ', x0)
        print('x1: ', x1)
        print('y0: ', y0)
        print('y1: ', y1)

    #default subplot values
    p_x0 = plt.rcParams['figure.subplot.left']
    p_x1 = plt.rcParams['figure.subplot.right']
    p_y0 = plt.rcParams['figure.subplot.bottom'] 
    p_y1 = plt.rcParams['figure.subplot.top'] 

    
    #p_x0*fig_width_pt is the pt value of the axis origin
    #p_x0*fig_width_pt-ax_x0 is the shift to go from axis to labels
    new_p_x0 =(p_x0*fig_width_pt-x0+_buffer)/fig_width_pt
    new_p_x1 = 1.0 - (x1-p_x1*fig_width_pt+_buffer)/fig_width_pt
    
    new_p_y0 =(p_y0*fig_height_pt-y0+_buffer)/fig_height_pt
    new_p_y1 =1-(_buffer+y1-p_y1*fig_height_pt)/fig_height_pt
    #new_p_y1 = p_y1

    if axis == 'x':
        plt.gcf().subplots_adjust(left=new_p_x0)
        plt.gcf().subplots_adjust(right=new_p_x1)

        if debug:
            print('px0: ', new_p_x0)
            print('px1: ', new_p_x1)

    elif axis == 'y':

        plt.gcf().subplots_adjust(bottom=new_p_y0)
        plt.gcf().subplots_adjust(top=new_p_y1)

        if debug:
            print(p_y1*fig_height_pt)
            print('py0: ', p_y0, new_p_y0)
            print('py1: ', p_y1, new_p_y1)
    else:
        raise RuntimeError(f'Axis does not exist {axis}')


def plot_tightbox(fig, axes, fig_width_pt=None,fig_height_pt=None):
    if fig_width_pt is None:
        size = fig.get_size_inches()
        fig_width_pt = size[0]*fig.dpi
        fig_height_pt = size[1]*fig.dpi

    axes = axes.flatten()

    #find min, max values
    x0, x1, y0, y1 = [], [], [], []
    for ax in axes:
        bbox = ax.get_tightbbox(
            fig.canvas.get_renderer(),       
            call_axes_locator = True,  
            bbox_extra_artists = None
        )

        ax_x0 = bbox.xmin
        ax_y0 = bbox.ymin
        ax_x1= bbox.xmax
        ax_y1= bbox.ymax

        x0.append(ax_x0)
        x1.append(ax_x1)
        y0.append(ax_y0)
        y1.append(ax_y1)

    ax_x0 = np.min(x0)
    ax_x1 = np.max(x1)

    ax_y0 = np.min(y0)
    ax_y1 = np.max(y1)
    
    
    ax = fig.add_axes([0,0,1,1])
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_zorder(1000)
    ax.patch.set_alpha(0.5)
    #ax.patch.set_color('r')
    
    x0 = ax_x0/fig_width_pt
    y0 = ax_y0/fig_height_pt
    w0 = (ax_x1 - ax_x0)/fig_width_pt
    h0 = (ax_y1-ax_y0)/fig_height_pt
    rect = patches.Rectangle((x0,y0),w0,h0,linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
