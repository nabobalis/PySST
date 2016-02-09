sunkit-sst
==========

[![Coverage Status](https://coveralls.io/repos/github/nabobalis/sunkit-sst/badge.svg?branch=master)](https://coveralls.io/github/nabobalis/sunkit-sst?branch=master)
[![Documentation Status](https://readthedocs.org/projects/sunkit-sst/badge/?version=latest)](http://sunkit-sst.readthedocs.org/?badge=latest)

This is a small Python module that is designed to allow the reading of SST cube files that are created from the Swedish Solar Telescopes.
This should also work for some IRIS data cubes, as they are also created with this method.

Install
-------

To install, it requires NumPy and SunPy to be installed.
These are also hard dependencies of the module as well.

Currently, to install, you have to use the setup.py, like so:

```bash
sudo python setup.py install
```

Example
-------

```python
from sunkitsst import read_cube

im_file = 'path to imcube file'
sp_file = 'path to spcube file'

im_header, im_cube, sp_header, sp_cube = read_cubes(im_file, sp_file)

```
