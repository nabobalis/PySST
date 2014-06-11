from __future__ import absolute_import, division
from sunkitsst.sstmap import get_header_item_group, SSTMap
import sunpy.map as smap
from sunkitsst.read_cubes import read_cubes
from sunkitsst.visualisation import cube_explorer
import numpy as np
import matplotlib.pyplot as plt
import glob

plt.ion()
smap.Map.register(SSTMap, SSTMap.is_source_for)

imfile = '/data/SST/fastrbe/sstdata.icube'
spfile = '/data/SST/fastrbe/sstdata.sp.icube'
im_header, outmemmap, sp_header, sp_cube = read_cubes(imfile, spfile,  memmap = True)

files = glob.glob("/data/Mounted/SWAT/fastrbe/sst2sdo/fits/sst/halpha/*.fits")
files.sort()

first_maps = smap.Map(files[0])
cadence = 2.195 #s
x = get_header_item_group(first_maps.meta, 'lpos_')
x.sort()
waves = list(zip(*x)[1])
waves.sort()
axis_range = [np.arange(0,cadence*outmemmap.shape[0],cadence), waves] + [first_maps.yrange] + [first_maps.xrange]

fig = plt.figure(figsize=(16,14))
moose = cube_explorer.PlotInteractor(outmemmap, first_maps.meta['cdelt1'], '/home/nabobalis/Dropbox/SavedSlits/',
                           axis_range=None, cmap='Greys_r', fig=fig, colorbar=True)