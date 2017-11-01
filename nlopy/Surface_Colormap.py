# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 10:03:53 2017

@author: Ethan Crowell

Contains a wrapper for doing 2D color maps
"""

from matplotlib import cm

def surface_colormap(f, cmap=cm.viridis, display=True, save=False):
    """This is just a simple wrapper for plot_surface(). I can never remember the
    setup."""
    
    plt.imshow(f, cmap=cmap)
    if display:
        plt.show()
    if save:
        name = input('Enter desired name of image (no spaces):')
        plt.savefig(name+'.png')