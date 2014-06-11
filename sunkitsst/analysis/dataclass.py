"""
"""

from __future__ import division
import numpy as np
#from astropy import io
import GUI.read_cubes as read_cubes
#import statsmodels.tsa.tsatools as tsa
#import scipy.optimize as optimize
import matplotlib.pyplot as plt
from os.path import expanduser
import scipy.fftpack as fft
#import scipy.signal as signal
import matplotlib.animation as anim
from matplotlib.image import NonUniformImage
import pycwt as wavelet

def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

class CubeDataAnalysis:
    """
    A analysis class for 4D data.

    Parameters
    ----------
    data: np.ndarray
        Array shapped as such [t,lambda,x,y]

    period: floar
        time step for data

    pixel_scale: float
        Pixel scale for spatial axes

    save_dir: string
        dir to save files to

    """

    def __init__(self, data, pixel_arc, period, savedir=None):

        self.pixel_arc = pixel_arc
        self.period = period
        if savedir:
            self.home = savedir
        else:
            self.home = expanduser("~")

        self.im_cube = data
        self.time = np.linspace(0,self.im_cube.shape[0]*self.period,self.im_cube.shape[0])

        self.extent= [0, self.pixel_arc*self.im_cube[0,0,:,:].shape[0],
                      0, self.pixel_arc*self.im_cube[0,0,:,:].shape[1]]

    def find_closest(self, array, target):
        """
        Finds the closest value in an array to a given target.
        Was taken from stackoverflow.

        Parameters
        ----------

        array : array_like
            Input array, should be real
        target : float or int
            Target value to locate.

        Returns
        -------

        idx : float
            The index where the nearst value is.

        """

        idx = array.searchsorted(target) #array must be sorted
        idx = np.clip(idx, 1, len(array)-1)
        left = array[idx-1]
        right = array[idx]
        idx -= target - left < right - target
        return idx

    def peek(self, wavelength):
        """
        Small function to look quickly at the data given.

        Parameters
        ----------

        wavelength : int or bool
            Wavelength to look at if given a non-singular wavelegth data cube.

        """
        fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(nrows=2, ncols=4)

        steps = np.linspace(0, self.im_cube.shape[0]-1,8)
        for i, ax in enumerate((ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8)):
            im = ax.imshow(self.im_cube[steps[i],wavelength], origin = 'lower', interpolation = 'nearest', extent=self.extent)
            im.set_cmap('gray')
        plt.show()

    def intensity_limit(self, background_box, bounding_box, lim, wavelength,
                        plot=False):
        """
        Finds the intensity limit of the data given due to an assumption that the data is normal.
        Then locates the area that is below this limit. To be used for finding dark structure's area.

        Paramters
        ---------

        bounding_box : list
            A 4 element list that conatining x0,x1,y0,y1 of the box.
        wavelength : int or bool
            Wavelength to look at if given a non-singular wavelength data cube.
        lim : float or int
            Limit for the area threshold.
        plot : bool
            Will plot the data if true for a quick look.

        Returns
        -------

        area : ndarray
            A 1D array with the area
        inten : ndarray
            A 1D array with the intensity
        lim_inten : ndarray
            A 1D array with the intensity limit at each time step.

        """

        area = np.zeros([self.im_cube.shape[0]])
        inten = np.zeros([self.im_cube.shape[0]])

        bound = self.im_cube[:,wavelength,bounding_box[2]:bounding_box[3],bounding_box[0]:bounding_box[1]]
        cut_box = self.im_cube[:,wavelength,background_box[2]:background_box[3],background_box[0]:background_box[1]]
        extent = [bounding_box[0]*self.pixel_arc,bounding_box[1]*self.pixel_arc,
                  bounding_box[2]*self.pixel_arc,bounding_box[3]*self.pixel_arc]

        if plot:
            def update_fig(i):
                im.set_array(cut_box[i])
                return im,

            def update_fig_1(i):
                im_1.set_array(bound[i])
                return im,

            fig = plt.figure()
            im = plt.imshow(cut_box[0], origin = 'lower', interpolation = 'nearest', cmap=plt.get_cmap('Greys'))
            ani = anim.FuncAnimation(fig, update_fig, interval=1, frames = range(0,int(cut_box.shape[0]) -1), blit=False)
            plt.show()

            fig_1 = plt.figure()
            im_1 = plt.imshow(bound[0], origin = 'lower', interpolation = 'nearest', cmap=plt.get_cmap('Greys'))
            ani_1 = anim.FuncAnimation(fig_1, update_fig_1, interval=1, frames = range(0,int(bound.shape[0]) -1), blit=False)
            plt.show()

        cut = cut_box.reshape(cut_box.shape[0],cut_box.shape[1]*cut_box.shape[2])
        lim_inten = np.mean(cut, axis = 1) - lim*np.std(cut, axis = 1)

        pore = np.zeros(bound.shape,dtype=np.int)

        ##TODO: SCIKIT IMAGE?
        for i in range(0,bound.shape[0]):
            pore[i] = (bound[i] <= lim_inten[i])
        for k in range(0,bound.shape[0]):
            area[k] = len(pore[k].nonzero()[0])
            inten[k] = np.sum(bound[k][pore[k].nonzero()])

        if plot:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
            im = ax1.imshow(bound[0], origin = 'lower', interpolation = 'nearest', extent=extent)
            ax1.contour(bound[0] <= lim_inten[0], origin = 'lower', interpolation = 'nearest', extent=extent, levels=[0,1])
            im.set_cmap('gray')
            ax2.hist(cut_box[0].flatten(),color='green', alpha=1)
            ax2.hist(bound[0].flatten(),color='red', alpha=0.2)
            ax3.plot(area)
            ax4.plot(inten,'r')
            plt.show()

        return area, inten, lim_inten

    def fft_period(self, bounding_box, period, wavelength,
                   plot=False, total=False, save=False):
        """
        Finds the overal FFT power of a small region for certina periods.

        #TODO: Control the size of the guassian that is used to window the FFT in the frequency domain.
        #TODO: Add more functions for windows.
        #TODO: Have the return be able to cope with several periods and wavelengths.

        Paramters
        ---------

        bounding_box : list
            A 4 element list that conatining x0,x1,y0,y1 of the box.
        period : list
            A list (can be singular) of all periods to cycle over.
        wavelength : list
            Wavelength to look at if given a non-singular wavelength data cube. List even if singular.
        plot : bool
            Will plot the data if true for a quick look.
            Will animate the data if total is set to false.
        total : bool
            Will sum over the entire time series for the FFT power.
        save : bool
            Will save the animations according to wavelength index number and period.

        Returns
        -------

        nothing due to potential size of FFT arrays. All saved to disk.

        """
        for wave in wavelength:
            im_cube = self.im_cube[:,wave,bounding_box[2]:bounding_box[3],bounding_box[0]:bounding_box[1]]
            for per in period:
                memmap = np.memmap(self.home + '/data_%s_%s' %tuple((per, wave)) + '_memmap.npy', mode='w+',
                                   shape=(im_cube.shape[0],im_cube.shape[1],im_cube.shape[2]),dtype = np.complex64)
                freq = fft.fftfreq(im_cube.shape[0], d = self.period)
                wanted_freq = 1/per
                f_up = wanted_freq + 0.002
                f_dn = wanted_freq - 0.002
                fd     = (f_up-f_dn)/2
                de    = fd**2/np.log10(10.)

                window = np.exp(-(freq[0:freq.shape[0]]-wanted_freq)**2/de)
                window += window[::-1]

                for i in range(memmap.shape[1]):
                    for j in range(memmap.shape[2]):
                        memmap[:,i,j] = (fft.fft(im_cube[:,i,j])*window)

                if plot:
                    fig = plt.figure()
                    if total:
                        data = np.sum(np.abs(memmap[:,:,:]),axis=0)
                        data /= data.max()
                        plt.imshow(data, origin = 'lower', interpolation = 'nearest', extent=self.extent, cmap=plt.cm.cubehelix)
                        plt.colorbar()
                        if save:
                             plt.savefig(self.home + str((per, wave)) + '.png')
                        else:
                            plt.show()
                    else:
                        data =  np.abs(np.log10(memmap[:,:,:]))**2
                        im = plt.imshow(data[0], origin = 'lower', interpolation = 'nearest',
                                    extent=self.extent,cmap=plt.cm.cubehelix)

                        def update_fig(i):
                            i *= 5
                            im.set_array(np.log10(data[i]))
                            return im,
                        ani = anim.FuncAnimation(fig, update_fig, interval=1, frames = range(0,int(memmap.shape[0]/5) -1), blit=False)
                        if save:
                            ani.save((self.home + str((per, wave)) + str('animation.mp4')).encode(), fps=30, extra_args=['-vcodec', 'libx264'])
                        else:
                            plt.show()
            return

    def wavelet(self, signal, mother='morlet', plot=True):
        """
        Takes a 1D signal and perfroms a continous wavelet transform.

        Parameters
        ----------

        time: ndarray
            The 1D time series for the data
        data: ndarray
            The actual 1D data
        mother: string
            The name of the family. Acceptable values are Paul, Morlet, DOG, Mexican_hat
        plot: bool
            If True, will return a plot of the result.
        Returns
        -------

        Examples
        --------

        """
        sig_level = 0.95
        std2 = signal.std() ** 2
        signal_orig = signal[:]
        signal = (signal - signal.mean())/ signal.std()
        t1 = np.linspace(0,self.period*signal.size,signal.size)
        wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(signal,
                                                              self.period,
                                                              wavelet=mother, dj=1/100)
        power = (np.abs(wave)) ** 2
        period = 1/freqs
#        alpha, _, _ = wavelet.ar1(signal)
        alpha = 0.0
        ## (variance=1 for the normalized SST)
        signif, fft_theor = wavelet.significance(1.0, self.period, scales, 0, alpha,
                                significance_level=sig_level, wavelet=mother)
        sig95 = np.ones([1, signal.size]) * signif[:, None]
        sig95 = power / sig95

        glbl_power = std2 * power.mean(axis=1)
        dof = signal.size - scales
        glbl_signif, tmp = wavelet.significance(std2, self.period, scales, 1, alpha,
                               significance_level=sig_level, dof=dof, wavelet=mother)

        ## indices for stuff
        idx = self.find_closest(period,coi.max())

        ## Into minutes
        t1 /= 60
        period /= 60
        coi /= 60

        if plot:
            plt.figure(figsize=(12,12))

            ax = plt.axes([0.1, 0.75, 0.65, 0.2])
            ax.plot(t1, signal_orig-signal_orig.mean(), 'k', linewidth=1.5)

            extent = [t1.min(),t1.max(),0,max(period)]
            bx = plt.axes([0.1, 0.1, 0.65, 0.55], sharex=ax)
            im = NonUniformImage(bx, interpolation='nearest', extent=extent)
            im.set_cmap('cubehelix')
            im.set_data(t1, period[:idx], power[:idx,:])
            bx.images.append(im)
            bx.contour(t1, period[:idx], sig95[:idx,:], [-99,1], colors='w', linewidths=2, extent=extent)
            bx.fill(np.concatenate([t1, t1[-1:]+self.period, t1[-1:]+self.period,t1[:1]-self.period, t1[:1]-self.period]),
                    (np.concatenate([coi,[1e-9], period[-1:], period[-1:], [1e-9]])),
                    'k', alpha=0.3,hatch='x', zorder=100)
            bx.set_xlim(t1.min(),t1.max())

            cx = plt.axes([0.77, 0.1, 0.2, 0.55], sharey=bx)
            cx.plot(glbl_signif[:idx], period[:idx], 'k--')
            cx.plot(glbl_power[:idx], period[:idx], 'k-', linewidth=1.5)
            cx.set_ylim(([min(period), period[idx]]))
            plt.setp(cx.get_yticklabels(), visible=False)

            plt.show()
        return wave, scales, freqs, coi, power

    def cross_wavelet(self, signal_1, signal_2, mother='morlet', plot=True):

        signal_1 = (signal_1 - signal_1.mean()) / signal_1.std()    # Normalizing
        signal_2 = (signal_2 - signal_2.mean()) / signal_2.std()    # Normalizing

        W12, cross_coi, freq, signif = wavelet.xwt(signal_1, signal_2, self.period, dj=1/100, s0=-1, J=-1,
                                             significance_level=0.95, wavelet=mother,
                                             normalize=True)

        cross_power = np.abs(W12)**2
        cross_sig = np.ones([1, signal_1.size]) * signif[:, None]
        cross_sig = cross_power / cross_sig
        cross_period = 1/freq

        WCT, aWCT, corr_coi, freq, sig = wavelet.wct(signal_1, signal_2, self.period, dj=1/100, s0=-1, J=-1,
                                                sig=False,significance_level=0.95, wavelet=mother,
                                                normalize=True)

        cor_sig = np.ones([1, signal_1.size]) * sig[:, None]
        cor_sig = np.abs(WCT) / cor_sig
        cor_period = 1/freq

        angle = 0.5 * np.pi - aWCT
        u, v = np.cos(angle), np.sin(angle)


        t1 = np.linspace(0,self.period*signal_1.size,signal_1.size)

        ## indices for stuff
        idx = self.find_closest(cor_period,corr_coi.max())

        ## Into minutes
        t1 /= 60
        cross_period /= 60
        cor_period /= 60
        cross_coi /= 60
        corr_coi /= 60

        fig1, ax1 = plt.subplots(nrows=1,ncols=1, sharex=True, sharey=True, figsize=(12,12))
        extent_cross = [t1.min(),t1.max(),0,max(cross_period)]
        extent_corr =  [t1.min(),t1.max(),0,max(cor_period)]
        im1 = NonUniformImage(ax1, interpolation='nearest', extent=extent_cross)
        im1.set_cmap('cubehelix')
        im1.set_data(t1, cross_period[:idx], cross_power[:idx,:])
        ax1.images.append(im1)
        ax1.contour(t1, cross_period[:idx], cross_sig[:idx,:], [-99, 1], colors='k', linewidths=2, extent=extent_cross)
        ax1.fill(np.concatenate([t1, t1[-1:]+self.period, t1[-1:]+self.period,t1[:1]-self.period, t1[:1]-self.period]),
                (np.concatenate([cross_coi,[1e-9], cross_period[-1:], cross_period[-1:], [1e-9]])),
                'k', alpha=0.3,hatch='x')
        ax1.set_title('Cross-Wavelet')
#        ax1.quiver(t1[::3], cross_period[::3], u[::3, ::3],
#                  v[::3, ::3], units='width', angles='uv', pivot='mid',
#                  linewidth=1.5, edgecolor='k', headwidth=10, headlength=10,
#                  headaxislength=5, minshaft=2, minlength=5)
        ax1.set_ylim(([min(cross_period), cross_period[idx]]))
        ax1.set_xlim(t1.min(),t1.max())

        fig2, ax2 = plt.subplots(nrows=1,ncols=1, sharex=True, sharey=True, figsize=(12,12))
        fig2.subplots_adjust(right=0.8)
        cbar_ax_1 = fig2.add_axes([0.85, 0.05, 0.05, 0.35])
        im2 = NonUniformImage(ax2, interpolation='nearest', extent=extent_corr)
        im2.set_cmap('cubehelix')
        im2.set_data(t1, cor_period[:idx], np.log10(WCT[:idx,:]))
        ax2.images.append(im2)
        ax2.contour(t1, cor_period[:idx], cor_sig[:idx,:], [-99, 1], colors='k', linewidths=2, extent=extent_corr)
        ax2.fill(np.concatenate([t1, t1[-1:]+self.period, t1[-1:]+self.period,t1[:1]-self.period, t1[:1]-self.period]),
                (np.concatenate([corr_coi,[1e-9], cor_period[-1:], cor_period[-1:], [1e-9]])),
                'k', alpha=0.3,hatch='x')
        ax2.set_title('Cross-Correlation')
#        ax2.quiver(t1[::3], cor_period[::3], u[::3,::3], v[::3,::3],
#                   units='height', angles='uv', pivot='mid',linewidth=1.5, edgecolor='k',
#                   headwidth=10, headlength=10, headaxislength=5, minshaft=2, minlength=5)
        ax2.set_ylim(([min(cor_period), cor_period[idx]]))
        ax2.set_xlim(t1.min(),t1.max())
        fig2.colorbar(im2, cax=cbar_ax_1)

        plt.show()

        plt.figure(figsize=(12,12))
        im3= plt.imshow(np.rad2deg(aWCT), origin='lower',interpolation='nearest', cmap='seismic', extent=extent_corr)
        plt.fill(np.concatenate([t1, t1[-1:]+self.period, t1[-1:]+self.period,t1[:1]-self.period, t1[:1]-self.period]),
                (np.concatenate([corr_coi,[1e-9], cor_period[-1:], cor_period[-1:], [1e-9]])),
                'k', alpha=0.3,hatch='x')
        plt.ylim(([min(cor_period), cor_period[idx]]))
        plt.xlim(t1.min(),t1.max())
        plt.colorbar(im3)
        plt.show()


        return

    def fft_analysis(self, signal, per):
        mean = signal.mean()
        print 'Signal Mean', signal.mean()
        signal -= signal.mean()

        freq = fft.fftfreq(len(signal),per)
        period = 1/freq[:len(signal)/2]
        period /= 60 #Minutes

        fft_ = fft.fft(signal)/ signal.shape[0]

        #magnitude and phase
        mag=np.abs(fft_[1:len(signal)/2])
        #phase=np.rad2deg(np.arctan(fft_[2:len(signal)/2]))
        phase = np.angle(fft_[1:len(signal)/2], deg=True)

        #power spectrum
        plt.figure()
        plt.plot(freq[1:len(signal)/2],mag)
        #plt.xlabel('Period (Minutes)')
        plt.title('Power')


        #phase
        plt.figure()
        plt.plot(freq[1:len(signal)/2],phase)
        #plt.hist(phase, bins=180)
        plt.title('Phase')
        #plt.xlabel('Period (Minutes)')

        plt.show()
        # find max in power spectrum
        idx = np.argmax(mag)
        mxmag=np.max(mag)
        mxphase=phase[idx]
        mxperiod=period[idx]

        print 'Max Period and Max Phase and Max Mag'
        print mxperiod,mxphase, mxmag

        # amplitude seismology parameter
        print 'amplitude seismology parameter'
        print mxmag/mean

    def EMD(self):
        pass

class SlitDataAnalysis(CubeDataAnalysis):

    def __init__(self):
        pass

if __name__ == '__main__':
    import copy
#    imfile = '/data/SST/fastrbe/sstdata.icube'
#    spfile = '/data/SST/fastrbe/sstdata.sp.icube'
#    imfile = '/data/SST/fastrbe/sdodata.icube'
#    spfile = '/data/SST/fastrbe/sdodata.sp.icube'
    imfile = '/data/SST/fastrbe/crisp.6563.icube'
    spfile = '/data/SST/fastrbe/crisp.6563.sp.icube'
#    imfile = '/data/SST/fastrbe/halpha.+-1032.doppler.icube'
#    spfile = None
#    imfile = '/data/SST/fastrbe/crispex.6302.fullstokes_aligned.icube'
#    spfile = '/data/SST/fastrbe/crispex.6302.fullstokes_aligned.sp.icube'
#    imfile = '/data/SST/arlimb/sstdata.fcube'
#    spfile = '/data/SST/arlimb/spsstdata.fcube'
#    imfile = '/data/SST/arlimb/sdodata.fcube'
#    spfile = '/data/SST/arlimb/spsdodata.fcube'

    per = 2.19572
    pix_arc = 0.059
    convert_area = (725 * pix_arc) ** 2
    im_header, im_cube, sp_header, sp_cube = read_cubes.read_cubes(imfile, spfile,  memmap = True)

    p = CubeDataAnalysis(im_cube, pix_arc, per, savedir='/data/')

    period = [120,180,240,300,360,420,540]
    bounding_box = [400,700,500,700] ##SST Focused
    background_box = [100,400,100,400] ##SST Focued
#    bounding_box = [210,370,200,290] ##SDO Focused
#    background_box = [390,550,270,370] ##SDO Focued

#
    area, inten, lim_inten = p.intensity_limit(background_box=background_box,
                                               bounding_box = bounding_box,
                                               lim = 3,
                                               wavelength=0, plot=True)
#
#    fft_data = p.fft_period(bounding_box = background_box,
#                            period=[120,180,240,300,360,420,540],
#                            wavelength=[1], plot=True, total=True, save=True)

#    wave, scales, freqs, coi, power = p.wavelet(area[274:],plot=True)
#    W12 = p.cross_wavelet(area[274:], inten[274:], mother='morlet',plot=True)
#                            pro fourier,data,time_res


#    crop_area = copy(area[300:1300])#*p.pixel_arc**2*725**2
#    crop_inten = copy(inten[300:1300])

#    p.wavelet(copy(crop_area))
#    p.wavelet(copy(crop_inten))

#    p.cross_wavelet(copy(crop_area), copy(crop_inten),
#                    mother='morlet',plot=True)

#    p.fft_analysis(copy(crop_area[9*60/per:22*60/per]),per)