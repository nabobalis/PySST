"""
Author: Staurt Mumford

A series of functions to read SST/SD0 data cubes created by some strage pipeline.
"""
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import numpy as np


def get_SST_header(afile):
    """Takes a SST cube file handle and returns a header dictionary"""
    afile.seek(0) # Assumes only one cube per file
    sheader = afile.read(512)
    header = {}
    for s in filter(None,sheader.decode("utf-8").split('\x00')):
        for s in s.strip().replace(':',',').split(', '):
            x = s.strip().split('=')
            try:
                header.update({x[0]:x[1]})
            except IndexError:
                pass
            else:
                header.update({x[0]:x[1]})

    #Convert cube dimensions to integers
    header['dims'] = int(header['dims'])
    header['ny'] = int(header['ny'])
    header['nx'] = int(header['nx'])

    if 'diagnostics' in header:
        header['diagnostics'] = list(header['diagnostics'].replace('[','').replace(']','').split(','))

    if header['dims'] != 3:
        raise Exception("Not 3D")
    else:
        header['nt'] = int(header['nt'])

    return header

def get_dtype(header):
    """Takes in a SST cube header and returns the correct numpy data type"""
    sdtype = header['datatype']
    endian = header['endian']

    if endian == 'l':
        np_endian = '<'
    elif endian == 'b':
        np_endian = '>'
    else:
        raise NotImplementedError("Big or Little Endian only implemented")

    #Get numeric datatype from header
    sdtype = int(sdtype.split()[0])

    if sdtype == 1:
        np_dtype = np.unit8
    elif sdtype == 2:
        np_dtype = '%s%s'%(np_endian,'i2')
    elif sdtype == 3:
        np_dtype = '%s%s'%(np_endian,'i4')
    elif sdtype == 4:
        np_dtype = '%s%s'%(np_endian,'f4')
    else:
        raise ValueError("Are you sure this header is valid?")

    return np.dtype((np_dtype))

def get_SST_cube(afile, header, np_dtype, memmap=True):
    """Given a SST data cube, a header dictionary and a computed numpy dtype
       returns a np array containg the whole data cube"""

    if memmap:
        data = np.memmap(afile, dtype=np_dtype, mode='r', order='C', offset=512,
                           shape=(header['nt'], header['ny'], header['nx']))
    else:
        afile.seek(512)
        count = header['ny'] * header['nx'] * header['nt']
        data = np.fromfile(afile, dtype=np_dtype, count=count)
        data = data.reshape(header['nt'], header['ny'], header['nx'])

    return data

def read_cubes(imfile, spfile=False, memmap = True):
    """High level  read, takes file name returns header and data cube"""
    im = open(imfile,'rb')
    im_header = get_SST_header(im)
    im_np_dtype = get_dtype(im_header)
    im_cube = get_SST_cube(im, im_header, im_np_dtype, memmap=memmap)

    if not(spfile):
        time = 1
        sp_header = False
        sp_cube = False
        im_cube = im_cube[:,None,...]
        sp_header = {'nx':1,'ny':im_cube.shape[0],'nt':im_cube.shape[-1]*im_cube.shape[-2]}
    else:
        sp = open(spfile,'rb')
        sp_header = get_SST_header(sp)
        sp_np_dtype = get_dtype(sp_header)
        sp_cube = get_SST_cube(sp, sp_header, sp_np_dtype, memmap=True)
        time = sp_header['nx']
    im_cube = np.reshape(im_cube, (im_header['nt'] / time,
                                   time, im_header['ny'],
                                            im_header['nx']) )

    return im_header, im_cube, sp_header, sp_cube