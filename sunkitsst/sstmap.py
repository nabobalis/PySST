from __future__ import print_function, unicode_literals, absolute_import, division

import sunpy.map
import sunpy.wcs as wcs

class SSTMap(sunpy.map.GenericMap):
    @property
    def date(self):
        return "{0}T{1}".format(self.meta['date'],self.meta['time'])

    @property
    def observatory(self):
        return self.meta['origin']

    @property
    def yrange(self):
        """Return the Y range of the image in arcsec from edge to edge."""
        ymin = self.center['y'] - self.shape[2] / 2. * self.scale['y']
        ymax = self.center['y'] + self.shape[2] / 2. * self.scale['y']
        return [ymin, ymax]

    @property
    def center(self):
        """Returns the offset between the center of the Sun and the center of
        the map."""
        return {'x': wcs.get_center(self.shape[1], self.scale['x'],
                                    self.reference_pixel['x'],
                                    self.reference_coordinate['x']),
                'y': wcs.get_center(self.shape[2], self.scale['y'],
                                    self.reference_pixel['y'],
                                    self.reference_coordinate['y']),}

    @classmethod
    def is_source_for(cls, data, header):
        if header['origin'].find('SST') != -1:
            return True
sunpy.map.Map.register(SSTMap, SSTMap.is_source_for)

def get_header_item_group(header, group):
    """
    Filter header and return list of items of a specific header
    group (e.g. 'CTYPE').
    Return empty list when unable to find @group in @_header.items().
    """
    return [i for i in header.items() if not i[0].find(group) and
                                               not i[0] == group]
