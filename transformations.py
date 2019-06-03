# -*- coding: utf-8 -*-
"""
Created on Sun Jun 2 2019
@name    Transformation Functions
@author: Jack Kriby Cook

"""

from abc import ABC
from collections import namedtuple as ntuple
import xarray as xr

import utilities.arrays as arrs

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['Normalize', 'Standardize', 'MinMax', 'Average', 'Cumulate', 'Consolidate', 'Interpolate', 'Inversion']
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""


Axis = ntuple('Axis', 'index key')

class Transformation(ABC):
    def __init__(self, xarrayfunc, *args, **hyperparms): 
        self.__hyperparms = hyperparms    
        self.__xarrayfunc = xarrayfunc
        
    def __call__(self, table, *args, axis, **kwargs):
        TableClass = table.__class__
        axis = Axis(table.axisindex(axis), table.axiskey(axis))
        xarray = xr.apply_ufunc(self.__xarrayfunc, table.xarray, *args, kwargs={'axis':-1, **self.__hyperparms, **kwargs},
                                input_core_dims=[[axis.key]], output_core_dims=[[axis.key]], vectorize=True, keep_attrs=True)
        specs = table.specs
        return TableClass(xarray, specs=specs, name=table.name)


class Normalize(Transformation):
    def __init__(self, *args, **kwargs): super().__init__(arrs.normalize, *args, **kwargs)  


class Standardize(Transformation):
    def __init__(self, *args, **kwargs): super().__init__(arrs.standardize, *args, **kwargs)


class MinMax(Transformation):
    def __init__(self, *args, **kwargs): super().__init__(arrs.minmax, *args, **kwargs)


class Average(Transformation):
    def __init__(self, *args, **kwargs): super().__init__(arrs.average, *args, **kwargs)


class Cumulate(Transformation):
    def __init__(self, *args, direction, **kwargs): super().__init__(arrs.cumulate, *args, direction=direction, **kwargs)    


class Consolidate(Transformation):
    pass


class Interpolate(Transformation): 
    def __init__(self, *args, fill, kind, **kwargs): super().__init__(arrs.interp1d, *args, fill=fill, kind=kind, invert=False, **kwargs)    


class Inversion(Transformation): 
    def __init__(self, *args, fill, kind, **kwargs): super().__init__(arrs.interp1d, *args, fill=fill, kind=kind, invert=True, **kwargs)  











