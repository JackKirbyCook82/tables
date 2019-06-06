# -*- coding: utf-8 -*-
"""
Created on Sun Jun 2 2019
@name    Transformation Functions
@author: Jack Kriby Cook

"""

from abc import ABC, abstractmethod
from collections import namedtuple as ntuple
import pandas as pd

import utilities.arrays as arr
import variables.arrays as var

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['Normalize', 'Standardize', 'MinMax', 'Average', 'Cumulate', 'Consolidate', 'Interpolate', 'Inversion']
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""


Axis = ntuple('Axis', 'index key')

class Transformation(ABC):
    def __init__(self, *args, functions={}, **hyperparms): 
        self.__hyperparms = hyperparms                
        self.__functions = functions
        
    def __call__(self, table, *args, axis, **kwargs):
        TableClass = table.__class__
        axis = Axis(table.axisindex(axis), table.axiskey(axis))
        xarray, specs = self.execute(table.xarray, table.specs, *args, axiskey=axis.key, **kwargs)
        return TableClass(xarray, specs=specs, name=table.name)
        
    def update_xarray(self, xarray, *args, axiskey, **kwargs):
        if 'xarray' in self.__functions.keys(): xarray = arr.apply_toarray(xarray, self.__functions['xarray'], *args, axis=axiskey, **self.__hyperparms, **kwargs)
        else: pass
        return xarray
        
    def update_header(self, xarray, *args, axiskey, **kwargs):
        if 'headers' in self.__functions.keys():
            headers = var.apply_tovariables(xarray.coords[axiskey].values, self.__functions['header'] *args, asstr=True, **self.__hyperparms, **kwargs)
            xarray.coords[axiskey] = pd.Index(headers, name=axiskey)
        else: pass
        return xarray

    @abstractmethod
    def execute(self, xarray, specs, *args, axiskey, **kwargs): pass


class Normalize(Transformation):
    def __init__(self, *args, **kwargs): 
        super().__init__(*args, functions={'xarray':arr.normalize}, **kwargs)  


class Standardize(Transformation):
    def __init__(self, *args, **kwargs): 
        super().__init__(*args, functions={'xarray':arr.standardize}, **kwargs)


class MinMax(Transformation):
    def __init__(self, *args, **kwargs): 
        super().__init__(*args, functions={'xarray':arr.minmax}, **kwargs)


class Average(Transformation):
    def __init__(self, *args, **kwargs): 
        super().__init__(*args, functions={'xarray':arr.average, 'headers':var.average}, **kwargs)


class Cumulate(Transformation):
    def __init__(self, *args, direction, **kwargs): 
        super().__init__(*args, functions={'xarray':arr.cumulate, 'headers':var.cumulate}, direction=direction, **kwargs)    


class Consolidate(Transformation):
    def __init__(self, *args, **kwargs): 
        super().__init__(*args, functions={'headers':var.consolidate}, **kwargs)    


class Interpolate(Transformation): 
    def __init__(self, *args, fill, kind, **kwargs): 
        super().__init__(*args, functions={'xarray':arr.interp1d}, fill=fill, kind=kind, invert=False, **kwargs)    


class Inversion(Transformation): 
    def __init__(self, *args, fill, kind, **kwargs): 
        super().__init__(*args, functions={'xarray':arr.interp1d}, fill=fill, kind=kind, invert=True, **kwargs)  












