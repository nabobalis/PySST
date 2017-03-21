from __future__ import (print_function, unicode_literals,
                        absolute_import, division)

import itertools
import glob
import datetime

import numpy as np
from matplotlib import widgets
import matplotlib.pyplot as plt
from sunpy.visualization.imageanimator import ImageAnimator
from sunkitsst.visualisation.slit import Slit

__all__ = ['PlotInteractor']


class PlotInteractor(ImageAnimator):
    """
    A PlotInteractor.
    
    Takes 4D (Time,Lambda,X,Y) or 5D (Time,Stokes,Lambda,X,Y) arrays

    Parameters
    ----------
    data: np.ndarray
        A 4D or 5D array

    pixel_scale: float
        Pixel scale for spatial axes

    save_dir: string
        dir to save slit files to

    axis_range: list or ndarray
        [min, max] pairs for each image axis and [min, max] pairs or arrays
        of values for each slider axis.
        Otherwise it just takes the shape and returns a non-physical index.
    """
    def __init__(self, data, pixel_scale, cadence, interop, savedir, **kwargs):
        all_axes = list(range(data.ndim))
        image_axes = [all_axes[i] for i in kwargs.get('image_axes', [-2, -1])]
        self.slider_axes = list(range(data.ndim))
        for x in image_axes:
            self.slider_axes.remove(x)

        if 'cmap' not in kwargs:
            kwargs['cmap'] = plt.get_cmap('gray')

        if data.ndim == 5:
            self.nlambda = data.shape[2]
            self.range = range(0, data.shape[2])
            axis_range = [None, [0, 10], None,
                      [0, pixel_scale * data[0, 0, 0, :, :].shape[0]],
                      [0, pixel_scale * data[0, 0, 0, :, :].shape[1]]]

        else:     
            self.nlambda = data.shape[1]
            self.range = range(0, data.shape[1])
            axis_range = [None, None,
                      [0, pixel_scale * data[0, 0, :, :].shape[0]],
                      [0, pixel_scale * data[0, 0, :, :].shape[1]]]

        axis_range = kwargs.pop('axis_range', axis_range)
        axis_range = self._sanitize_axis_range(axis_range, data)

        self.nt = data.shape[0]
        self.image_extent = list(itertools.chain.from_iterable([axis_range[i] for i in image_axes]))
        self.pixel_scale = pixel_scale
        self.cadence = cadence
        self.slits = []
        self.savedir = savedir
        self.interop = interop
        
        button_labels, button_func = self.create_buttons()

        slider_functions = [self._updateimage]*len(self.slider_axes) + [self.update_range]*2 + [self.update_im_clim]*2
        slider_ranges = [axis_range[i] for i in self.slider_axes] + [range(0, self.nlambda)]*2 + [np.arange(0, 99.9)]*2
        
        ImageAnimator.__init__(self, data, axis_range=axis_range,
                               button_labels=button_labels,
                               button_func=button_func,
                               slider_functions=slider_functions,
                               slider_ranges=slider_ranges,
                               **kwargs)

        # Sets up the slit sliders
        self.sliders[-4]._slider.set_val(self.nlambda)
        self.sliders[-3]._slider.slidermax = self.sliders[-4]._slider
        self.sliders[-4]._slider.slidermin = self.sliders[-3]._slider
        self.slider_buttons[-4].set_visible(False)
        self.slider_buttons[-3].set_visible(False)
        self.label_slider(-3, "Start")
        self.label_slider(-4, "End")

        # Sets up the intensity scaling sliders
        self.sliders[-2]._slider.set_val(100)
        self.sliders[-1]._slider.slidermax = self.sliders[-2]._slider
        self.sliders[-2]._slider.slidermin = self.sliders[-1]._slider
        self.slider_buttons[-1].set_visible(False)
        self.slider_buttons[-2].set_visible(False)
        self.axes.autoscale(False)
        self.label_slider(-1, "Min")
        self.label_slider(-2, "Max")

    def create_buttons(self):
        button_labels = ['Slit', 'Delete', 'Save', 'Load']
        button_func = [self.record, self.delete, self.save_slit, self.load_slit]

        return button_labels, button_func

    def update_im_clim(self, val, im, slider):
        if np.mean(self.data[self.frame_slice]) < 0:
            self.im.set_clim(np.min(self.data[self.frame_slice]) * (self.sliders[-1]._slider.val / 100),
                             np.max(self.data[self.frame_slice]) * (self.sliders[-2]._slider.val / 100))
        else:
            self.im.set_clim(np.max(self.data[self.frame_slice]) * (self.sliders[-1]._slider.val / 100),
                             np.max(self.data[self.frame_slice]) * (self.sliders[-2]._slider.val / 100))

    def update_range(self, val, im, slider):
        self.range = np.arange(int(self.sliders[3]._slider.val),int(self.sliders[2]._slider.val))
        if len(self.range) == 0:
            self.range = np.arange(int(self.sliders[3]._slider.val)-1,int(self.sliders[2]._slider.val)+1,1)

# =============================================================================
# Button Functions
# =============================================================================

    def delete(self, event):
        if not hasattr(self.slit, 'mpl_points'):
            print('You have not yet generated a curve to delete.')
        else:
            if len(self.slit.mpl_points) > 0 and len(self.slit.mpl_curve) > 0:
                self.slit.remove_all(self.slits)
                self.cid = None
                self.slits = []
        if hasattr(self, 'cursor'):
            self.fig.canvas.mpl_disconnect(self.cid)
            del self.cursor

    def record(self, event):
        if event.inaxes is None:
            return
        self.slit = Slit(self.axes, self.pixel_scale)
        self.slits.append(self.slit)
        self.cursor = widgets.Cursor(self.axes, useblit=False, color='red', linewidth=1)
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.get_click)

    def save_slit(self, event, filename=False):
        if not hasattr(self.slit, 'mpl_points'):
            print('There is no slit to save.')
        else:
            names = ['curve_points', 'slit_data', 'length']
            if not filename:
                filename = str(datetime.datetime.now())
                np.savez(self.savedir + filename, names, self.slit.curve_points, self.slit.data, self.slit.length)

    def load_slit(self, event):
        files = glob.glob(self.savedir + '*.npz')

        if len(files) <= 0:
            print('There seems to be no save files in this directory.')
            return

        for i in range(len(files)):
            name = files[i]
            self.slit = Slit(self.axes, self.pixel_scale)
            self.slits.append(self.slit)
            data = np.load(name).items()
            self.slit.data = data[0][1]
            self.slit.curve_points = data[1][1]
            self.slit.length = data[2][0]
            self.slit.mpl_curve.append(self.axes.plot(self.slit.curve_points[:, 0], self.slit.curve_points[:, 1]))
            self.axes.figure.canvas.draw()
            slit = np.zeros([len(self.range), self.nt, self.slit.res])
            for i, idx in enumerate(self.range):
                slit[i, :, :] = self.slit.get_slit_data(self.data[:, idx, :, :], self.image_extent)                
            slit = self.slit.get_slit_data(self.data[:, self.sliders[1]._slider.cval, :, :], self.image_extent)
            self.slit.length *= self.pixel_scale
            self.slit.data = slit
            self.plot_slits(slit)

# =============================================================================
# Figure Callbacks
# =============================================================================

    def get_click(self, event):
        if event.inaxes is not None:
            if event.inaxes is self.axes and event.button == 1:
                self.slit.add_point(event.xdata, event.ydata)
            elif event.inaxes is self.axes and event.button == 3:
                self.slit.remove_point()
            elif event.inaxes is self.axes and event.button == 2:
                self.slit.create_curve(self.interop)
                slit = np.zeros([len(self.range), self.nt, self.slit.res])
                for i, idx in enumerate(self.range):
                    slit[i, :, :] = self.slit.get_slit_data(self.data[:, idx, :, :], self.image_extent)
                self.slit.length *= self.pixel_scale
                self.slit.data = slit
                self.plot_slits(slit)
                self.fig.canvas.mpl_disconnect(self.cid)
                self.cid = None
            else:
                print('Click a real mouse button')

    def plot_slits(self, slit):
        extent = [0, self.nt*self.cadence, 0, self.slit.length]
        fig, axes = plt.subplots(nrows=slit.shape[0], ncols=1,
                                 sharex=True, sharey=True, figsize=(10, 18))
        if slit.shape[0] == 1:
            axes = [axes]

        for i in range(slit.shape[0]):
            loc_mean = slit[i, :, :].T/np.max(np.abs(slit[i, :, :].T))
            axes[i].imshow(loc_mean[:, :], origin='lower', interpolation='nearest',
                           cmap=plt.get_cmap('Greys_r'), extent=extent, aspect='auto')
            axes[i].set_xlim(0, extent[1])
            axes[i].set_ylim(0, extent[3])
        plt.xlabel('Time (seconds)')
        plt.ylabel('Length along slit (arcsecs)')
        fig.tight_layout()
        fig.subplots_adjust(hspace=0, wspace=0)
        fig.show()
