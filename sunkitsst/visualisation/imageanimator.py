# -*- coding: utf-8 -*-

from sunpy.visualization.imageanimator import ImageAnimator
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import matplotlib.animation as mplanim

from mpl_toolkits.axes_grid1 import make_axes_locatable
import mpl_toolkits.axes_grid1.axes_size as Size

__all__ = ['ImageAnimatorSST']


class ImageAnimatorSST(ImageAnimator):
    __doc__ = ImageAnimator.__doc__

    def _parse_axis_range(self, axis_range, data):
        """
        Slider axes should be a list of data.shape[i] long, and image axes should
        be [min,max] pairs for imshow
        """
        #If no axis range at all make it all [min,max] pairs
        if axis_range is None:
            axis_range = [[0, i] for i in data.shape]

        #need the same numer of axis ranges as axes
        if len(axis_range) != data.ndim:
            raise ValueError("axis_range must equal number of axes")

        #For each axis validate and translate the axis_range
        for i,d in enumerate(data.shape):
            #If [min,max] pair or None
            if len(axis_range[i]) == 2 or axis_range[i] is None:
                #If min==max or None
                if axis_range[i][0] == axis_range[i][1] or axis_range[i] is None:
                    if i in self.slider_axes:
                        axis_range[i] = np.linspace(0,d,d)
                    else:
                        axis_range[i] = [0, d]

            #If we have a whole list of values for the axis, make sure we are a slider axis.
            elif len(axis_range[i]) == d:
                if i not in self.slider_axes:
                    raise ValueError("Slider axes mis-match, non-slider axes need [min,max] pairs")
                else:
                    #Make sure the resulting element is a ndarray
                    axis_range[i] = np.array(axis_range[i])

            #panic
            else:
                raise ValueError("axis_range should be either: None, [min,max], or a linspace for slider axes")
        return axis_range

    def _updateimage(self, val, im, slider):
        ax = self.slider_axes[slider.slider_ind]
        ind = np.argmin(np.abs(self.axis_range[ax] - val))
        self.frame_slice[ax] = ind
        if val != slider.cval:
            im.set_array(self.data[self.frame_slice])
            slider.cval = val