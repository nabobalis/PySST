sunkit-sst
==========

This is a small Python module that is designed to allow the reading of SST cube files that are created from the Swedish Solar Telescopes.
This should also work for some IRIS datascubes, as they are also created with this method.

Install
-------

To install, it requires numpy and sunpy to be installed.
These are also hard dependacies of the module as well.

Currently, to install, you have to use the setup.py, like so:

```bash
sudo python setup.py install
```

Example
------

```python
from sunkitsst import read_cube

im_file = 'path to imcube file'
sp_file = 'path to spcube file'

im_header, im_cube, sp_header, sp_cube = read_cubes(im_file, sp_file)

```