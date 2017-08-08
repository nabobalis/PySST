File Format
===========
The file format for these SST cubes is as follows. Note that this is gleamed from the header and having reverse enginered the file format.

Each file has a header that is 512 bytes long and then the rest is binary data.
The shape of the binary data varies between the icube and spcube, however, it is the same data as far as I can tell.

The shape of the icube data is (nx,ny,bt).

The shape of the spcube data is ().

CRISPEX uses this file for a spectral view. However, I cannot see why. If anyone knows it would be very helpful.
These shortnames are explained in the Header section below.

Icubeheader : nx, stokes, endian, dims, datatype, ns, nt, and ny.
spcube header: nx, dims, ny, datatype, endian, and nt.
Headers are 512 bytes long for each cube.
If a cube has only one wavelength, the head in has nx for the sp cube as 4. This could be a mistake in the reduction pipeline of the data I have. Thus this could be fixed in the latest pipeline.
for the spcube file ny is time and nt is nx times ny.
Header
------

icube header:

- nx
- stokes
- endian
- dims
- datatype
- ns
- nt
- ny

spcube header:

- nx
- dims
- ny
- datatype
- endian
- nt

