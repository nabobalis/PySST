from __future__ import (print_function, unicode_literals,
                        absolute_import, division)

import scipy.ndimage as ndimage
import scipy.interpolate as interpolate
import numpy as np

__all__ = ['Slit']

# =============================================================================
# Slit Class
# =============================================================================


class Slit:

    def __init__(self, image_animator, pixel_scale):
        self.image_animator = image_animator
        self.pixel_scale = pixel_scale
        self.axes = self.image_animator.axes
        self.points = []
        self.mpl_points = []
        self.mpl_curve = []
        self.anns = []
        self.data = []
        self.data_run = []
        self.length = 0

    def add_point(self, x, y):
        self.points.append([x, y])
        self.mpl_points.append(self.axes.scatter(x, y))
        self.anns.append(self.axes.annotate('{}'.format(len(self.mpl_points)),
                                            (x + 1, y + 1)))
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

    def create_curve(self, interop):
#        self.curve_points = np.zeros([self.res, 2])
        self.t = np.arange(self.points[0][0],self.points[-1][0],self.pixel_scale)#linspace(0, self.res, self.res) / self.res
        self.res = self.t.shape[0]        
        if len(self.mpl_points) == 2:
            if interop:
                self.curve_points = self.interpol(*self.points)
            else:
                self.curve_points = self.linear_bezier(*self.points)
        elif len(self.mpl_points) == 3:
            if interop:
                self.curve_points = self.interpol(*self.points)
            else:
                self.curve_points = self.quad_bezier(self.points[0], self.points[-1], self.points[1])
        elif len(self.mpl_points) == 4:
            if interop:
                self.curve_points = self.interpol(*self.points)
            else:
                self.curve_points = self.cubic_bezier(self.points[0], self.points[2], self.points[-1], self.points[1])
        else:
            self.curve_points = self.interpol(*self.points)
        self.mpl_curve.append(self.axes.plot(self.curve_points[:, 0], self.curve_points[:, 1]))
        self.axes.figure.canvas.draw()

    def interpol(self, *args):
        x, y = zip(*args)
        if len(x) == 2:
            k = 1
        elif len(x) == 3:
            k = 2
        else:
            k = 3
        tck, u = interpolate.splprep([x, y], k=k)
        unew = np.linspace(0, self.res, self.res) / self.res
        curve = interpolate.splev(unew, tck)
        ans = np.zeros([self.res, 2])
        ans[:, 0] = curve[0]
        ans[:, 1] = curve[1]
        return ans

    def cubic_bezier(self, P0, P1, P2, P3):
        ans = np.zeros([self.res, 2])
        ans[:, 0] = (1 - t)**3 * P0[0] + 3*(1 - t)**2 *t*P1[0] + 3*(1-t)*t**2*P2[0] + t**3*P3[0]
        ans[:, 1] = (1 - t)**3 * P0[1] + 3*(1 - t)**2 *t*P1[1] + 3*(1-t)*t**2*P2[1] + t**3*P3[1]
        return ans

    def quad_bezier(self, P0, P1, P2):
        ans = np.zeros([self.res, 2])
        ans[:, 0] = (1 - t)**2 * P0[0] + 2*(1 - t)*t*P1[0] + t**2*P2[0]
        ans[:, 1] = (1 - t)**2 * P0[1] + 2*(1 - t)*t*P1[1] + t**2*P2[1]
        return ans

    def linear_bezier(self, P0, P1):
        ans = np.zeros([self.res, 2])
        ans[:, 0] = (1 - t) * P0[0] + t*P1[0]
        ans[:, 1] = (1 - t) * P0[1] + t*P1[1]
        return ans

    def get_slit_data(self, data, extent, order=0, mode="nearest"):
        if not hasattr(self, 'curve_points'):
            print('You have not yet generated a curve.')

        x_pixel = (self.curve_points[:, 0] - extent[2] )/ ((extent[3] - extent[2]) / data.shape[2])
        y_pixel = (self.curve_points[:, 1] - extent[0] )/ ((extent[1] - extent[0]) / data.shape[2])

        dist_x = (x_pixel[:-1] - x_pixel[1:]) ** 2
        dist_y = (y_pixel[:-1] - y_pixel[1:]) ** 2
        self.length = np.sum(np.sqrt(dist_x + dist_y))

        if len(data.shape) == 2:
            slit = ndimage.interpolation.map_coordinates(data, [y_pixel, x_pixel], order=order,mode=mode)
        elif len(data.shape) == 3:
            slit = np.zeros([data.shape[0], self.res])
            for i in range(0, data.shape[0]):
                slit[i, :] = ndimage.interpolation.map_coordinates(data[i, :, :], [y_pixel, x_pixel], order=order)
        else:
            raise Exception
        return slit
