# -*- coding: utf-8 -*-
"""
Specific Case for 4D data (time, wavelength, two spatial axes)
"""
from __future__ import absolute_import
import glob
import datetime
import itertools

import numpy as np
import matplotlib.pyplot as plt

import scipy.ndimage as ndimage
import scipy.interpolate as interpolate
from matplotlib import widgets
from skimage import exposure

import sunpy.map
import sunpy.wcs as wcs
from sunpy.visualization.imageanimator import ImageAnimator


__all__ = ['PlotInteractor']

#==============================================================================
# Slit Class
#==============================================================================

class Slit:

    def __init__(self, axes):
        self.axes = axes
        self.points = []
        self.mpl_points = []
        self.mpl_curve = []
        self.anns = []
        self.res = 50
        self.curve_points = np.zeros([self.res,2])
        self.data = []
        self.data_run = []
        self.distance = []

    def add_point(self, x, y):
        self.points.append([x,y])
        self.mpl_points.append(self.axes.scatter(x,y))
        self.anns.append(self.axes.annotate('%i' % len(self.mpl_points), (x + 1, y + 1)))
        self.axes.figure.canvas.draw()

    def remove_point(self):
        if len(self.mpl_points) > 0:
            point = self.mpl_points.pop(-1)
            point.set_visible(False)
            annote = self.anns.pop(-1)
            annote.set_visible(False)
            self.points.pop(-1)
            self.axes.figure.canvas.draw()

    def remove_all(self, slits):
        if len(slits) > 0:
            for i in range(len(slits)):
                if len(slits[i].mpl_points) > 0:
                    for y in range(len(slits[i].mpl_points)):
                        point = slits[i].mpl_points.pop()
                        point.set_visible(False)
                        annote = slits[i].anns.pop()
                        annote.set_visible(False)
                        slits[i].points.pop()
                    if len(slits[i].mpl_curve) > 0:
                     for line in self.axes.lines:
                         self.axes.lines.pop()
            self.axes.figure.canvas.draw()

    def create_curve(self):
        flag = False # True - Beizer Curve else Interpolation
#        if self.mpl_points[0]-self.mpl_points[1]:
#            self.res = 150
#        elif b:
#            self.res = 150
#        elif c:
#            self.res = 150
#        else:
#            self.res = 150

#        self.curve_points = np.zeros([self.res,2])
        if len(self.mpl_points) == 2:
            self.curve_points = self.linear_bezier(*self.points)
        elif len(self.mpl_points) == 3:
            if flag:
                self.curve_points = self.quad_bezier(self.points[0],self.points[-1],self.points[1])
            else:
                self.curve_points = self.interpol(*self.points)
        elif len(self.mpl_points) == 4:
            if flag:
                self.curve_points = self.cubic_bezier(self.points[0],self.points[2],self.points[-1],self.points[1])
            else:
                self.curve_points = self.interpol(*self.points)
        else:
            self.curve_points = self.interpol(*self.points)
        self.mpl_curve.append(self.axes.plot(self.curve_points[:,0], self.curve_points[:,1]))
        self.axes.figure.canvas.draw()

    def interpol(self,*args):
        x,y = zip(*args)
        if len(x) == 3:
            k = 2
        else:
            k = 3
        tck,u = interpolate.splprep([x,y], k = k)
        unew = np.linspace(0, self.res,self.res) / self.res
        curve = interpolate.splev(unew,tck)
        ans = np.zeros([self.res, 2])
        ans[:,0] = curve[0]
        ans[:,1] = curve[1]
        return ans

    def cubic_bezier(self, P0, P1, P2, P3):
        ans = np.zeros([self.res, 2])
        t = np.linspace(0, self.res,self.res) / self.res
        ans[:,0] = (1 - t)**3 * P0[0] + 3*(1 - t)**2 *t*P1[0] + 3*(1-t)*t**2*P2[0] + t**3*P3[0]
        ans[:,1] = (1 - t)**3 * P0[1] + 3*(1 - t)**2 *t*P1[1] + 3*(1-t)*t**2*P2[1] + t**3*P3[1]
        return ans

    def quad_bezier(self, P0, P1, P2):
        ans = np.zeros([self.res, 2])
        t = np.linspace(0, self.res,self.res) / self.res
        ans[:,0] = (1 - t)**2 * P0[0] + 2*(1 - t)*t*P1[0] + t**2*P2[0]
        ans[:,1] = (1 - t)**2 * P0[1] + 2*(1 - t)*t*P1[1] + t**2*P2[1]
        return ans

    def linear_bezier(self, P0, P1):
        ans = np.zeros([self.res, 2])
        t = np.linspace(0, self.res,self.res) / self.res
        ans[:,0] = (1 - t) * P0[0] + t*P1[0]
        ans[:,1] = (1 - t) * P0[1] + t*P1[1]
        return ans

    def get_slit_data(self, data, extent, order=3):
        if not hasattr(self, 'curve_points'):
            print('You have not yet generated a curve.')

#        import pdb; pdb.set_trace()

        x_pixel = (self.curve_points[:,0] - extent[2] )/ ((extent[3] - extent[2]) / data.shape[2])
        y_pixel = (self.curve_points[:,1] - extent[0] )/ ((extent[1] - extent[0]) / data.shape[2])

        dist_x = (x_pixel[:-1] - x_pixel[1:]) ** 2
        dist_y = (y_pixel[:-1] - y_pixel[1:]) ** 2
        self.distance = np.sum(np.sqrt(dist_x + dist_y))

        if len(data.shape) == 2:
            slit = ndimage.interpolation.map_coordinates(data, [y_pixel,x_pixel], order=order)
        elif len(data.shape) == 3:
            slit = np.zeros([data.shape[0],self.res])
            for i in range(0,data.shape[0]):
                slit[i,:] = ndimage.interpolation.map_coordinates(data[i,:,:], [y_pixel,x_pixel], order=order)
        else:
            raise Exception
        return slit

    def get_run_diff(self, slit, sort='normal', j = 5):
        if sort == 'normal':
            self.data_run.append(slit[:-1] - slit[1:])
            return self.data_run[-1]
        elif sort == 'baseline':
            self.data_run.append(slit[:-1] - slit[0])
            return self.data_run[-1]
        elif sort == 'symmetric':
            self.data_run.append(slit[:-1] - slit[1:])
            return self.data_run[-1]
        elif sort == 'symmetric':
            self.data_run.append(slit[:-1] - slit[1:])
            return self.data_run[-1]
        elif sort == 'symmetric':
            self.data_run.append(slit[:-1] - slit[1:])
            return self.data_run[-1]


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
        [min, max] pairs for each axis
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

        axis_range = self._parse_axis_range(axis_range, data)

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
                                    aspect='auto')#, vmin=np.min(rundiff),
                                    #vmax=0.8*np.max(rundiff))
                axes[0].imshow(slit[i,:,:].T/np.max(np.abs(slit[i,:,:].T)), origin='lower',
                                    interpolation='spline36',
                                    cmap=plt.get_cmap('Greys_r'), extent = extent,
                                    aspect='auto')#, vmin=np.min(slit[i,:,:]),
                                    #vmax=0.8*np.max(slit[i,:,:]))
            else:
#                loc_mean = exposure.equalize_adapthist(slit[i,:,:].T/np.max(np.abs(slit[i,:,:].T)), clip_limit=0.15, nbins=2**12)
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


class SSTMap(sunpy.map.GenericMap):
    @property
    def date(self):
        return "{0}T{1}".format(self.meta['date'],self.meta['time'])

    @property
    def observatory(self):
        return self.meta['origin']

    @property
    def yrange(self):
        """Return the Y range of the image in arcsec from edge to edge."""
        ymin = self.center['y'] - self.shape[2] / 2. * self.scale['y']
        ymax = self.center['y'] + self.shape[2] / 2. * self.scale['y']
        return [ymin, ymax]

    @property
    def center(self):
        """Returns the offset between the center of the Sun and the center of
        the map."""
        return {'x': wcs.get_center(self.shape[1], self.scale['x'],
                                    self.reference_pixel['x'],
                                    self.reference_coordinate['x']),
                'y': wcs.get_center(self.shape[2], self.scale['y'],
                                    self.reference_pixel['y'],
                                    self.reference_coordinate['y']),}

    @classmethod
    def is_source_for(cls, data, header):
        if header['origin'].find('SST') != -1:
            return True
sunpy.map.Map.register(SSTMap, SSTMap.is_source_for)
