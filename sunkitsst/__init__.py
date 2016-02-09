# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
This is an Astropy affiliated package.
"""

# Affiliated packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._astropy_init import *
# ----------------------------------------------------------------------------

# For egg_info test builds to pass, put package imports here.
if not _ASTROPY_SETUP_:
        from sunkitsst.read_cubes import (get_SST_header, get_SST_cube,
                                  get_dtype, read_cubes)
        from sunkitsst.visualisation.cube_explorer import PlotInteractor
        from sunkitsst.visualisation.slit import Slit

__all__ = ["get_SST_header", "get_SST_cube", "get_dtype", "read_cubes",
           "PlotInteractor", "Slit"]
