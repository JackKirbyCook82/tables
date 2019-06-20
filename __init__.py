# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 2019
@name:   Table Objects
@author: Jack Kirby Cook

"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import json
from functools import update_wrapper

from tables.operations import OPERATIONS

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
    def __init__(self, *args, data, name='table', variables, **kwargs): 
        self.__tabledata = data
        self.__tablename = name
        self.__variables = variables
    
    @property
    def name(self): return self.__tablename
    @property
    def __name(self): return ' '.join([self.__tablename.upper(), 'as', self.__class__.__name__.upper()])
    @property
    def data(self): return self.__tabledata
    @property
    def variables(self): return self.__variables
    
    def __str__(self, *args, **kwargs): return '\n'.join([_buffer, '\n\n'.join([self.__name, *self.strings, 'Dim={}, Shape={}'.format(self.dim, self.shape)]), _buffer])        
    def __len__(self): return self.dim    
    
    @abstractmethod
    def strings(self): pass
    @abstractmethod
    def dim(self): pass
    @abstractmethod
    def shape(self): pass  
 

class ArrayTable(TableBase):
    def __init__(self, xarray, *args, **kwargs):
        self.__xarray = xarray
        super().__init__(*args, **kwargs)
    
    @property
    def xarray(self): return self.__xarray   
    def todict(self): return dict(xarray=self.xarray, data=self.data, variables=self.variables)
    
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

    def __add__(self, other): return self.add(other)
    def __sub__(self, other): return self.subtract(other)
    def __mul__(self, other): return self.multiply(other)
    def __truediv__(self, other): return self.divide(other)
    
    def __getattr__(self, attr): 
        try: operation_function = OPERATIONS[attr]
        except KeyError: raise AttributeError('{}.{}'.format(self.__class__.__name__, attr))
        
        def wrapper(other, *args, **kwargs): return operation_function(self, other, *args, **kwargs)
        update_wrapper(wrapper, operation_function)   
        return wrapper
        
    def flatten(self): 
        pass        


class FlatTable(TableBase):
    def __init__(self, dataframe, *args, **kwargs):
        self.__dataframe = dataframe
        super().__init__(*args, **kwargs)

    @property
    def dataframe(self): return self.__dataframe
    @property
    def variables(self): return self.__variables

    @property
    def dim(self): pass
    @property
    def shape(self): pass

    @property
    def strings(self):
        pass

    def unflatten(self):
        pass
    
    
    
    
    
    
    
    
    
    
    
    
    
    