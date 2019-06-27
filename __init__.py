# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 2019
@name:   Table Objects
@author: Jack Kirby Cook

"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from functools import update_wrapper

from utilities.dataframes import dataframe_fromxarray
from utilities.xarrays import xarray_fromdataframe
from utilities.strings import uppercase

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


_buffer = lambda : _OPTIONS['bufferchar'] * _OPTIONS['linewidth']
_aslist = lambda items: [items] if not isinstance(items, (list, tuple)) else list(items)


class TableBase(ABC):
    variableformat = 'VARIABLE[{index}] = {key}: {values}' 
    
    def __init__(self, data, key, name, variables, **kwargs): 
        self.__data, self.__key, self.__name = data, key, name
        self.__variables = variables.__class__({key:value for key, value in variables.items() if key in self.items()})
    
    @property
    def data(self): return self.__data
    @property
    def key(self): return self.__key   
    @property
    def name(self): return self.__name
    @property
    def variables(self): return self.__variables
    
    @property
    def tablename(self): return ' '.join([self.__name.upper(), 'as', self.__class__.__name__.upper()])    
    def todict(self): return dict(data=self.data, key=self.key, name=self.name, variables=self.variables)
    
    def __getitem__(self, key): return self.__class__(self.data[key], key=self.key, name=self.name, variables=self.variables)
    def __str__(self): return '\n'.join([_buffer(), '\n\n'.join([self.tablename, self.strings, self.variablestrings, 'Dim={}, Shape={}'.format(self.dim, self.shape)]), _buffer()])        
    def __len__(self): return self.dim    
    
    @property
    def variablestrings(self): return '\n'.join([self.variableformat.format(index=index, key=uppercase(key), values=values.name()) for index, key, values in zip(range(len(self.variables)), self.variables.keys(), self.variables.values())])  
    
    @abstractmethod
    def strings(self): pass
    @abstractmethod
    def dim(self): pass
    @abstractmethod
    def shape(self): pass  

    @abstractmethod
    def items(self): pass
 

class ArrayTable(TableBase):
    dataformat = 'DATA = {key}:\n{values}'
    headerformat = 'HEADER[{index}] = {key}:\n{values}'
    scopeformat = 'SCOPE[{index}] = {key}: {values}'    
    
    @property
    def xarray(self): return self.data   
    
    @property
    def dim(self): return len(self.xarray.dims)
    @property
    def shape(self): return self.xarray.shape 
        
    def axiskey(self, axis): return self.xarray.dim[axis] if isinstance(axis, int) else axis
    def axisindex(self, axis): return self.xarray.get_axis_num(axis) if isinstance(axis, str) else axis
    def items(self): return [self.key, *self.xarray.attrs.keys(), *self.xarray.coords.keys()]
    
    @property
    def strings(self): return '\n\n'.join([self.datastring, self.headerstrings, self.scopestrings])
    @property
    def datastring(self): return self.dataformat.format(key=uppercase(self.key), values=self.xarray.values)
    @property
    def headerstrings(self): return '\n'.join([self.headerformat.format(index=index, key=uppercase(axis), values=self.xarray.coords[axis].values) for index, axis in zip(range(len(self.xarray.dims)), self.xarray.dims)])
    @property
    def scopestrings(self): return '\n'.join([self.scopeformat.format(index=index, key=uppercase(key), values=values) for index, key, values in zip(range(len(self.xarray.attrs)), self.xarray.attrs.keys(), self.xarray.attrs.values())])

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
        return FlatTable(dataframe_fromxarray(self.xarray, self.key), key=self.key, name=self.name, variables=self.variables)     


class FlatTable(TableBase):
    dataformat = 'DATA = {key}:\n{values}'
    
    @property
    def dataframe(self): return self.data

    @property
    def dim(self): return len(self.headercolumns)
    @property
    def shape(self): return tuple([len(set(self.dataframe[column].values)) for column in self.headercolumns])

    @property
    def strings(self): return self.dataformat.format(key=uppercase(self.key), values=str(self.data))

    @property
    def datacolumns(self): return [self.key]
    @property
    def scopecolumns(self): return [column for column in self.dataframe.columns if len(set(self.dataframe[column].values)) == 1]
    @property
    def headercolumns(self): return [column for column in self.dataframe.columns if all([column not in self.scopecolumns, column != self.key])]
    def items(self): return self.datacolumns + self.scopecolumns + self.headercolumns
    
    def unflatten(self):
        return ArrayTable(xarray_fromdataframe(self.dataframe, key=self.key), key=self.key, name=self.name, variables=self.variables)
    
    def createdata(self, key, *args, axes, function, varfunction=None, **kwargs):
        dataframe, variables = self.dataframe, self.variables
        dataframe[key] = dataframe[_aslist(axes)].apply(function, axis=1, *args, **kwargs)
        variables[key] = varfunction(*[variables[axis] for axis in axes], *args, **kwargs)
        return self.__class__(dataframe, key=self.key, name=self.name, variables=variables)


class GeoTable(object):
    def __init__(self, data, name, repository, **kwargs): 
        self.__data, self.__name = data, name
        self.__repository = repository
    
    @property
    def tablename(self): return ' '.join([self.__name.upper(), 'as', self.__class__.__name__.upper()])   
    def __str__(self): return '\n'.join([_buffer(), self.tablename, str(self.__data), _buffer()])
    

    
    
    
    
    
    