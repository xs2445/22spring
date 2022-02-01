"""
This file contains animation utility functions for displaying convolution animations.

E6692 Spring 2022

YOU DO NOT NEED TO MODIFY THESE FUNCTIONS TO COMPLETE THE ASSIGNMENT
"""

from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from matplotlib import animation


def animate(i, n_conv, output_signal, conv_mask, l2, l3, l4):
    """
    animate update the data for the FuncAnimation design
    
    Args:
        i (int): number of frame visualized
        n_conv (int): length of conv_mask
        output_signal (list): list of the applied convolution
        conv_mask (list): list of the conv_mask elements
        l2 (Line2D): dataset for the visualization of conv mask
        l3 (Line2D): dataset for the output signal (points)
        l4 (Line2D): dataset for the output signal (line)
    """
    left = len(conv_mask) // 2
    right = len(conv_mask) - left - 1
    l2.set_data(range(i - left - 1, i + right), conv_mask)
    l3.set_data(range(0, i), [output_signal[x] for x in range(0, i)])
    l4.set_data(range(0, i), [output_signal[x] for x in range(0, i)])
    
    
def visualize_1dcond(input_signal, output_signal, conv_mask, conv=True):
    """
    visualize_1dcond creates a jshtml object to visualizes the a 1d convolution 
    with matplotlib.animation.FuncAnimation
    
    The startframe is -1 (no convolution applied)
    
    Args:
        input_signal (list): list of the input_signla elements
        output_signal (list): list of the applied convolution
        conv_mask (list): list of the conv_mask elements
    
    Returns:
        ani (jshtml): Return the animation as jshtml
    
    Todo:
        * Making parameters available to the function and not hardcoded
    """
    n_conv = len(conv_mask)
    n_input_signal = len(input_signal)
    
    # Calculate the boundaries of the plot
    min_x = -n_conv
    max_x = len(input_signal)+n_conv
    min_y = min(output_signal)-1
    max_y = min(output_signal)+1
    
    # Initialize the plot
    fontP = FontProperties()
    fontP.set_size('small')
    plt.rcParams['figure.figsize'] = [8, 4]
    fig, ax = plt.subplots(nrows=2, ncols=1)
    if conv:
        fig.suptitle('Visualization conv1d')
    else:
        fig.suptitle('Visualization correlation')
    ax[0].axis([min_x, max_x, -1, max(output_signal)+1])
    ax[0].set_ylabel('Signal')
    ax[1].axis([min_x, max_x, -1, max(output_signal)+1])
    ax[1].set_ylabel('Signal')
    ax[1].set_xlabel('Step')
    
    ## input signal
    l, = ax[0].plot(range(0, n_input_signal), input_signal, label='input')
    ## convmask
    if conv:
        l2, = ax[0].plot([0], [0], 'r', label='convmask')
    else:
        l2, = ax[0].plot([0], [0], 'r', label='corrmask')
    ax[0].legend(prop=fontP)
    ## output signal as points
    l3, = ax[1].plot([0], [0], 'bo', label='output')
    ## output signal as line
    l4, = ax[1].plot([], [], 'b')
    ax[0].xaxis.set_ticklabels([])
    ax[1].legend(prop=fontP)
    plt.subplots_adjust(hspace = 0.05)
    
    # Initialize animation
    ani = animation.FuncAnimation(fig, animate, frames=n_input_signal+1, fargs=(n_conv, output_signal, conv_mask, l2, l3, l4))
    
    # Run animation
    return(ani.to_jshtml())