PySST
==========

[![Build Status](https://travis-ci.org/nabobalis/pysst.svg?branch=master)](https://travis-ci.org/nabobalis/pysst)
[![Documentation Status](http://readthedocs.org/projects/pysst/badge/?version=latest)](http://docs.sunpy.org/projects/pysst/en/latest/?badge=latest)

This is a small Python module that is designed to allow the reading of SST cube files that are created from the Swedish Solar Telescopes.
This should also work for some IRIS data cubes, as they are also created with this method.

Furthermore, there is a basic matplotlib visualization tool that allows a basic flick through the data and do a slit analysis.

Install
-------

To install, it requires NumPy and SunPy to be installed.
These are also hard dependencies of the module as well.

Currently, to install, you have to use the setup.py, like so:

```bash
pip install -e .
```

Example
-------

```python
from pysst import read_cube

im_file = 'path to imcube file'
sp_file = 'path to spcube file'

im_header, im_cube, sp_header, sp_cube = read_cubes(im_file, sp_file)

```

The documentation, linked above, goes into more detail about the strucure of the cubes as well as the other features of this module.
