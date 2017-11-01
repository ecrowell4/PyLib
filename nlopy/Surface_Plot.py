# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 10:03:53 2017

@author: Ethan Crowell

Contains a wrapper for doing 3d plots with plot_surface()
"""

from matplotlib import cm

def surface_plot(x, y, f, cmap=cm.viridis, rstride=1, cstride=1,
                 display=True, save=False):
    """This is just a simple wrapper for plot_surface()."""
    
    fig = plt.figure()
    
    # These two lines I can never remember
    ax = fig.gca(projection='3d')
    
    # Make the actual plot
    surf = ax.plot_surface(X, Y, f, rstride=1, cstride=1, cmap=cmap, shade=True)
    
    # The user can choose to display and/or save the plot.
    if display:
        plt.show()
    if save:
        # Prompt user for desire name of image
        name = input('Enter desired name of image (no spaces):')
        plt.savefig(name+'.png')
