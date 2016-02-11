Welcome to Sunkit-SST's documentation!
======================================

This is a small Python module that allows the reading of icube and spcube files that are created by the reduction pipeline from the Swedish Solar Telescope (SST).

Furthermore, there is a basic visualisation module that uses Matplotlib to create a GUI. From this, it is possible to browse through the data and do a slit analysis. It is not intended to be a replacement for `CRISPEX <http://folk.uio.no/gregal/crispex/>`_.

**Please note that this was created for SST cubes created in 2012 and has not been tested on cubes created from the current SST reduction pipeline.**

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

.. automodapi:: sunkitsst
