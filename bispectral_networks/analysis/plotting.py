from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from IPython.display import HTML



def image_grid(data, shape=(10,10), figsize=(10,10), cmap='Greys_r', share_range=True, interpolation=None, save_name=None):

    fig = plt.figure(figsize=figsize)
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=shape,  # creates 10x10 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )

    if share_range:
        vmin = data.min()
        vmax = data.max()
        
    for ax, im in zip(grid, data):
        # Iterating over the grid returns the Axes.
        if share_range:
            ax.imshow(im, vmin=vmin, vmax=vmax, cmap=cmap, interpolation=interpolation)
        else:
            ax.imshow(im, cmap=cmap, interpolation=interpolation)
        ax.set_axis_off()
        
    if save_name is not None:
        plt.savefig(save_name)
        
        
def animated_video(vid, interval=25, figsize=(5,5), cmap='Greys_r'):
    def init():
        return (im,)

    def animate(frame):
        im.set_data(frame)
        return (im,)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(np.zeros(vid[0].shape), vmin=np.min(vid), vmax=np.max(vid), cmap=cmap);
    plt.axis('off')
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=vid, interval=interval, blit=True)
    plt.close()
    return HTML(anim.to_jshtml())

