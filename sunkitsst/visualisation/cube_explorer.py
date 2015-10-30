from __future__ import print_function, unicode_literals, absolute_import, division

import itertools
import numpy as np
from matplotlib import widgets
import glob
import datetime
import matplotlib.pyplot as plt

from sunpy.visualization.imageanimator import ImageAnimator
from sunkitsst.visualisation.slit import Slit
__all__ = ['PlotInteractor']

#==============================================================================
# Plot Class
#==============================================================================

class PlotInteractor(ImageAnimator):
    """
    A PlotInteractor.
    t,lambda,x,y

    Parameters
    ----------
    data: np.ndarray
        A 4D array

    pixel_scale: float
        Pixel scale for spatial axes

    save_dir: string
        dir to save slit files to

    axis_range: list or ndarray
        [min, max] pairs for each image axis and [min, max] pairs or arrays
        of values for each slider axis.
    """
    def __init__(self, data, pixel_scale, savedir, **kwargs):
        all_axes = list(range(data.ndim))
        image_axes = [all_axes[i] for i in kwargs.get('image_axes', [-2,-1])]
        self.slider_axes = list(range(data.ndim))
        for x in image_axes:
            self.slider_axes.remove(x)

        axis_range = [None,None,
                      [0, pixel_scale * data[0,0,:,:].shape[0]],
                      [0, pixel_scale * data[0,0,:,:].shape[1]]]
        axis_range = kwargs.pop('axis_range', axis_range)

        axis_range = self._sanitize_axis_range(axis_range, data)

        self.image_extent = list(itertools.chain.from_iterable([axis_range[i] for i in image_axes]))
        self.pixel_scale = pixel_scale
        self.r_diff = []
        self.slits = []
        self.savedir = savedir
        self.nlambda = data.shape[1]
        self.nt = data.shape[0]

        button_labels, button_func = self.create_buttons()

        slider_functions = [self._updateimage]*len(self.slider_axes) + [self.update_im_clim]*2
        slider_ranges = [axis_range[i] for i in self.slider_axes] + [np.arange(0,99.9)]*2

        ImageAnimator.__init__(self, data, axis_range=axis_range,
                                  button_labels=button_labels,
                                  button_func=button_func,
                                  slider_functions=slider_functions,
                                  slider_ranges=slider_ranges,
                                  **kwargs)

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
#==============================================================================
# Button Functions
#==============================================================================

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
        if event.inaxes is None: return
        self.slit = Slit(self.axes)
        self.slits.append(self.slit)
        self.cursor = widgets.Cursor(self.axes, useblit=False, color='red', linewidth=1)
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.get_click)

    def save_slit(self, event, filename=False):
        if not hasattr(self.slit, 'mpl_points'):
            print('SAVE BEN FOGLE, SAVE THE SLIT.')
        else:
            names = ['curve_points', 'slit_data', 'distance']
            if not filename:
                filename = str(datetime.datetime.now())
            if self.r_diff:
                np.savez(self.savedir + filename, names + ['run_diff'],
                         self.slit.curve_points, self.slit.data, self.slit.distance, self.slit.data_run)
            else:
                np.savez(self.savedir + filename, names,
                         self.slit.curve_points, self.slit.data, self.slit.distance)

    def load_slit(self, event):
        files_npz = glob.glob(self.savedir + '*.npz')
        files_npy = glob.glob(self.savedir + '*.npy')

        if len(files_npz) > 0 and len(files_npy) == 0:
            files = files_npz
            flag = 'npz'
        elif len(files_npy) > 0 and len(files_npz) == 0:
            files = files_npy
            flag = 'npy'
        else:
            print('Needs work and needs Ben Fogle')
            return

        for i in range(len(files)):
            name = files[i]
            self.slit = Slit(self.axes)
            self.slits.append(self.slit)
            if flag == 'npz':
                data = np.load(name).items()
                self.slit.data = data[0][1]
                self.slit.curve_points = data[1][1]
#                self.slit.distance = data[2][0]
            elif flag == 'npy':
                self.slit.curve_points[:,0], self.slit.curve_points[:,1] = zip(*np.load(name))
            self.slit.mpl_curve.append(self.axes.plot(self.slit.curve_points[:,0], self.slit.curve_points[:,1]))
            self.axes.figure.canvas.draw()
            slit = np.zeros([self.nlambda,self.nt,self.slit.res])
            for i in range(self.nlambda):
                slit[i,:,:] = self.slit.get_slit_data(self.data[:,i,:,:],self.image_extent)
            self.slit.distance *= self.pixel_scale
            self.slit.data = slit
            self.plot_slits(slit)

#==============================================================================
# Figure Callbacks
#==============================================================================

    def get_click(self, event):
        if not event.inaxes is None:
            if event.inaxes is self.axes and event.button == 1:
                self.slit.add_point(event.xdata,event.ydata)
            elif event.inaxes is self.axes and event.button == 3:
                self.slit.remove_point()
            elif event.inaxes is self.axes and event.button == 2:
                self.slit.create_curve()
                slit = np.zeros([self.nlambda,self.nt,self.slit.res])
                for i in range(self.nlambda):
                    slit[i,:,:] = self.slit.get_slit_data(self.data[:,i,:,:],self.image_extent)
#                profiler.stop()
#                print(profiler.output_text(unicode=True, color=True))
                self.slit.distance *= self.pixel_scale
                self.slit.data = slit
                self.plot_slits(slit)
                self.fig.canvas.mpl_disconnect(self.cid)
                self.cid = None
            else:
                print('Click a real mouse button')

    def plot_slits(self, slit, r_diff=False):
        extent = [0, self.nt, 0 , self.slit.distance]
        self.r_diff = r_diff

        if r_diff:
            fig, axes = plt.subplots(nrows=self.nlambda, ncols=2,
                                     sharex=True, sharey=True, figsize = (10,8))
        else:
            fig, axes = plt.subplots(nrows=self.nlambda, ncols=1,
                                     sharex=True, sharey=True, figsize = (6,9))
        if self.nlambda == 1 and not r_diff:
            axes = [axes]

        for i in range(0, self.nlambda):
            if r_diff:
                rundiff = self.slit.get_run_diff(slit[i,:,:])
                axes[1].imshow(rundiff[:,:].T/np.max(np.abs(rundiff[:,:].T)), origin='lower',
                                interpolation='spline36',
                                 cmap=plt.get_cmap('Greys_r'), extent = extent,
                                    aspect='auto')
                axes[0].imshow(slit[i,:,:].T/np.max(np.abs(slit[i,:,:].T)), origin='lower',
                                    interpolation='spline36',
                                    cmap=plt.get_cmap('Greys_r'), extent = extent,
                                    aspect='auto')
            else:
                loc_mean = slit[i,:,:].T/np.max(np.abs(slit[i,:,:].T))

                axes[i].imshow(loc_mean[:,:], origin='lower',
                                interpolation='spline36',
                                cmap=plt.get_cmap('Greys_r'), extent = extent,
                                    aspect='auto')
                axes[i].set_xlim(0,extent[1])
                axes[i].set_ylim(0,extent[3])
        fig.tight_layout()
        fig.subplots_adjust(hspace=0, wspace=0)
        fig.show()
