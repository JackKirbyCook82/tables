# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 2019
@name:   Table Objects
@author: Jack Kirby Cook

"""

from abc import ABC, abstractmethod
import xarray as xr
import pandas as pd
import numpy as np
import json

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['ArrayTable', 'FlatTable']
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""


_OPTIONS = dict(linewidth=100, maxrows=30, maxcolumns=10, threshold=100, precision=3, bufferchar='=')
_PDMAPPING = {"display.max_rows":"maxrows", "display.max_columns":"maxcolumns"}
_NPMAPPING = {"linewidth":"linewidth", "threshold":"threshold", "precision":"precision"}

def apply_options():
    for key, value in _PDMAPPING.items(): pd.set_option(key, _OPTIONS[value])
    np.set_printoptions(**{key:_OPTIONS[value] for key, value in _NPMAPPING.items()})

def set_options(*args, **kwargs):
    global _OPTIONS
    _OPTIONS.update(kwargs)
    apply_options()
    
def get_option(key): return _OPTIONS[key]
def show_options(): print('Table Options: ' + ', '.join([' = '.join([key, str(value)]) for key, value in _OPTIONS.items()]))


_buffer = _OPTIONS['bufferchar'] * _OPTIONS['linewidth']


class TableBase(ABC):
    def __init__(self, *args, name='table', **kwargs): self.__tablename = name
    def __str__(self, *args, **kwargs): return '\n'.join([_buffer, '\n\n'.join([self.__name, *self.strings, 'Dim={}, Shape={}'.format(self.dim, self.shape)]), _buffer])        
    @property
    def name(self): return self.__tablename
    @property
    def __name(self): return ' '.join([self.__tablename.upper(), 'as', self.__class__.__name__.upper()])
    def __len__(self): return self.dim    
    
    @abstractmethod
    def strings(self): pass
    @abstractmethod
    def dim(self): pass
    @abstractmethod
    def shape(self): pass  
 

class ArrayTable(TableBase):
    def __init__(self, xarray, *args, specs={}, **kwargs):
        assert isinstance(xarray, xr.DataArray)
        self.__xarray = xarray
        self.__specs = specs
        super().__init__(*args, **kwargs)
    
    @property
    def xarray(self): return self.__xarray
    @property
    def specs(self): return self.__specs
    
    @property
    def dim(self): return len(self.xarray.dims)
    @property
    def shape(self): return self.xarray.shape 
    
    def axiskey(self, axis): return self.__xarray.dim[axis] if isinstance(axis, int) else axis
    def axisindex(self, axis): return self.__xarray.get_axis_num(axis) if isinstance(axis, str) else axis
    
    @property
    def strings(self):
        data = self.__xarray.values
        headers = json.dumps({dim:list(self.__xarray.coords[dim].values) for dim in self.__xarray.dims}, sort_keys=False, indent=3, separators=(',', ' : '))
        scope = json.dumps(self.__xarray.attrs, sort_keys=False, indent=3, separators=(',', ' : '))
        return ['DATA\n' + str(data), 'HEADERS\n' + str(headers), 'SCOPE\n' + str(scope)]
    
    def flatten(self): 
        pass
    

class FlatTable(TableBase):
    def __init__(self, dataframe, *args, specs={}, **kwargs):
        assert isinstance(dataframe, pd.DataFrame)
        self.__dataframe = dataframe
        self.__specs = specs
        super().__init__(*args, **kwargs)

    @property
    def dataframe(self): return self.__dataframe
    @property
    def specs(self): return self.__specs

    @property
    def dim(self): pass
    @property
    def shape(self): pass

    @property
    def strings(self):
        pass

    def unflatten(self, datakey):
        pass
    
    
    
    
    
    
    
    
    
    
    
    
    
    