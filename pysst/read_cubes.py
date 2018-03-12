# -*- coding: utf-8 -*-
"""
A series of functions to read SST data cubes created by the reduction pipeline.
"""
from __future__ import (print_function, unicode_literals, absolute_import, division)

import numpy as np
import warnings

# Number of Stoke profiles IQUV
NUM_STOKES = 4


def get_SST_header(afile):
    """
    Takes a SST cube file handle and returns a header dictionary

    Parameters
    ----------
    afile: open file instance
        An open file instance to the SST cube file.

    Returns
    -------
    header: dictonary
        The dictonary that contains the header of the SST cube file.
    """
    afile.seek(0)  # Assumes only one cube per file
    sheader = afile.read(512)
    header = {}
    for s in filter(None, sheader.decode("utf-8").split('\x00')):
        for s in s.strip().replace(':', ',').split(', '):
            x = s.strip().split('=')
            try:
                header.update({x[0]: x[1]})
            except IndexError:
                pass
            else:
                header.update({x[0]: x[1]})

    #  Convert cube dimensions to integers
    header['dims'] = int(header['dims'])
    header['ny'] = int(header['ny'])
    header['nx'] = int(header['nx'])

    if 'ns' in header.keys():
        header['ns'] = int(header['ns'])

    if 'diagnostics' in header:
        di = header['diagnostics'].replace('[', '').replace(']', '').split(',')
        header['diagnostics'] = list(di)

    if header['dims'] != 3:
        raise Exception("Not 3D")
    else:
        header['nt'] = int(header['nt'])

    return header


def get_dtype(header):
    """
    Takes in a SST cube header and returns the correct numpy data type

    Parameters
    ----------
    header: dictonary
        The dictonary that contains the header of the SST cube file.


    Returns
    -------
    np_dtype : type
        This is data type for the given SST cube header.
    """
    sdtype = header['datatype']
    endian = header['endian']

    if endian == 'l':
        np_endian = '<'
    elif endian == 'b':
        np_endian = '>'
    else:
        raise NotImplementedError("Big or Little Endian only implemented")

    # Get numeric datatype from header
    sdtype = int(sdtype.split()[0])

    if sdtype == 1:
        np_dtype = np.unit8
    elif sdtype == 2:
        np_dtype = '{}{}'.format(np_endian, 'i2')
    elif sdtype == 3:
        np_dtype = '{}{}'.format(np_endian, 'i4')
    elif sdtype == 4:
        np_dtype = '{}{}'.format(np_endian, 'f4')
    else:
        raise ValueError("Are you sure this header is valid?")

    return np.dtype((np_dtype))


def get_SST_cube(afile, header, np_dtype, memmap=True):
    """
    Given a SST data cube, a header dictionary and a computed numpy dtype
    returns a np array containg the whole data cube.


    Parameters
    ----------
    afile: string
        A filepath to the a SST cube file.
    header: dictonary
        The dictonary that contains the header of the SST cube file.
    np_dtype: type
        The data type of the data within the SST cube file.
    memmap: Bool
        If True, will use memmap (default) in order to save your RAM.


    Returns
    -------
    data : array
        This is data array for that SST cube file.
    """
    if memmap:
        data = np.memmap(
            afile,
            dtype=np_dtype,
            mode='r',
            order='C',
            offset=512,
            shape=(header['nt'], header['ny'], header['nx']))
    else:
        afile.seek(512)
        count = header['ny'] * header['nx'] * header['nt']
        data = np.fromfile(afile, dtype=np_dtype, count=count)
        data = data.reshape(header['nt'], header['ny'], header['nx'])

    return data


def read_cubes(imfile, spfile=False, memmap=True, n_wave=None):
    """
    High level  read, takes file name returns header and data cube

    Parameters
    ----------
    imfile: string
        A filepath to the imcube file.
    spfile: string
        A filepath to the spcube file.
    memmap : Bool
        If True, will use memmap (default) in order to save your RAM.
    n_wave : `int`
        If known (and not using ``spfile`` specify the number of wavelength
        point to reconstruct the full cube shape.

    Returns
    -------
    im_header : dictonary
        This is the header for the imcube file.
    im_cube: array
        This is the numpy array of the data from the imcube file.
    sp_header: dictonary
        This is the header for the spcube file.
    sp_cube: array
        This is the numpy array of the data from the spcube file.
    """
    im = open(imfile, 'rb')
    im_header = get_SST_header(im)
    im_np_dtype = get_dtype(im_header)
    im_cube = get_SST_cube(im, im_header, im_np_dtype, memmap=memmap)

    if not (spfile):
        time = 1
        sp_header = False
        sp_cube = False

        # This can be relaxed later
        if not n_wave:
            raise ValueError("Right now we need this input before hand.\
                              This will be relaxed in a new version.")

        if 'ns' in im_header.keys():
            # We have stokes
            n_l = im_header['ns'] * n_wave
            time = im_header['nt'] // n_l
            im_cube = im_cube[:, None, None, ...]
            target_shape = (np.int(im_header['ns']), np.int(time), np.int(n_wave),
                            np.int(im_header['ny']), np.int(im_header['nx']))
        else:
            time = im_header['nt'] / n_wave
            target_shape = (np.int(time), np.int(n_wave),
                            np.int(im_header['ny']), np.int(im_header['nx']))
            im_cube = im_cube[:, None, ...]

        sp_header = {'nx': 1, 'ny': im_cube.shape[0], 'nt': im_cube.shape[-1] * im_cube.shape[-2]}
    else:
        sp = open(spfile, 'rb')
        sp_header = get_SST_header(sp)
        sp_np_dtype = get_dtype(sp_header)
        sp_cube = get_SST_cube(sp, sp_header, sp_np_dtype, memmap=True)
        if 'ns' in im_header.keys():
            if im_header['ns']==1:
                target_shape = (np.int(sp_header['ny']), np.int(sp_header['nx']),
                                np.int(im_header['ny']), np.int(im_header['nx']))
            else:
                # 4 stoke paramaters
                target_shape = (np.int(sp_header['ny']), np.int(NUM_STOKES),
                                np.int(sp_header['nx']), np.int(im_header['ny']),
                                np.int(im_header['nx']))
                warnings.warn("Cube is shaped as Time, Stokes, Lambda, X, Y", UserWarning)
        else:
            target_shape = (np.int(sp_header['ny']), np.int(sp_header['nx']),
                            np.int(im_header['ny']), np.int(im_header['nx']))

    # TODO: Might be better not to reshape it this way.
    im_cube = np.reshape(im_cube, target_shape)

    return im_header, im_cube, sp_header, sp_cube
