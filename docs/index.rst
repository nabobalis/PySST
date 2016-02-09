Welcome to Sunkit-SST's documentation!
======================================

This is a small Python module that is designed to allow the reading of SST cube files that are created from the Swedish Solar Telescopes.
This should also work for some IRIS data cubes, as they are also created with this method.

Furthermore, there is a visualisation module that uses matplotlib to create a GUI.
From this, it is possible to flick through the data as well as do a slit analysis.

Install
-------

To install, it requires NumPy and SunPy to be installed.
These are also hard dependencies of the module as well.

Currently, to install, you have to use the setup.py, like so:

.. code-block:: bash

  sudo python setup.py install

Example
-------

This is the most basic example possible with this code.

.. code-block:: python

  from sunkitsst import read_cubes

  im_file = 'path to imcube file'
  sp_file = 'path to spcube file'

  im_header, im_cube, sp_header, sp_cube = read_cubes(im_file, sp_file)

API
---

Below are the functions included within this Python module.

.. autofunction:: sunkitsst.get_SST_header
.. autofunction:: sunkitsst.get_SST_cube
.. autofunction:: sunkitsst.get_dtype
.. autofunction:: sunkitsst.read_cubes
.. autofunction:: sunkitsst.cube_explorer
.. autofunction:: sunkitsst.slit

