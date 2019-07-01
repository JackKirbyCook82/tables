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
from utilities.dispatchers import clskey_singledispatcher as keydispatcher

from tables.operations import OPERATIONS

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['ArrayTable', 'FlatTable', 'GeoTable']
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
    def __init__(self, *args, data, name, **kwargs): 
        self.__data, self.__name = data, name
   
    @property
    def data(self): return self.__data
    @property
    def name(self): return self.__name
    
    @property
    def tablename(self): return ' '.join([self.__name.upper(), 'as', self.__class__.__name__.upper()])    
    
    def __str__(self): return '\n'.join([_buffer(), '\n\n'.join([self.tablename, self.strings, 'Dim={}, Shape={}'.format(self.dim, self.shape)]), _buffer()])        
    def __len__(self): return self.dim    
    
    @abstractmethod
    def strings(self): pass
    @abstractmethod
    def dim(self): pass
    @abstractmethod
    def shape(self): pass  

    @abstractmethod
    def todict(self): pass
    @abstractmethod
    def items(self): pass
 

class ArrayTable(TableBase):
    dataformat = 'DATA = {key}:\n{values}'
    headerformat = 'HEADER[{index}] = {key}:\n{values}'
    scopeformat = 'SCOPE[{index}] = {key}: {values}'    
    variableformat = 'VARIABLE[{index}] = {key}: {values}' 
    
    def __init__(self, *args, key, variables, **kwargs):
        super().__init__(*args, **kwargs)
        self.__key = key
        self.__variables = variables.__class__({key:value for key, value in variables.items() if key in self.items()})
           
    @property
    def xarray(self): return self.data   
    @property
    def key(self): return self.__key
    @property
    def variables(self): return self.__variables
    
    @property
    def dim(self): return len(self.xarray.dims)
    @property
    def shape(self): return self.xarray.shape 
        
    def axiskey(self, axis): return self.xarray.dim[axis] if isinstance(axis, int) else axis
    def axisindex(self, axis): return self.xarray.get_axis_num(axis) if isinstance(axis, str) else axis
    def items(self): return [self.key, *self.xarray.attrs.keys(), *self.xarray.coords.keys()]
    def todict(self): return dict(data=self.xarray, key=self.key, name=self.name, variables=self.variables)
    
    @property
    def strings(self): return '\n\n'.join([self.datastring, self.headerstrings, self.scopestrings, self.variablestrings])
    @property
    def datastring(self): return self.dataformat.format(key=uppercase(self.key, withops=True), values=self.xarray.values)
    @property
    def headerstrings(self): return '\n'.join([self.headerformat.format(index=index, key=uppercase(axis, withops=True), values=self.xarray.coords[axis].values) for index, axis in zip(range(len(self.xarray.dims)), self.xarray.dims)])
    @property
    def scopestrings(self): return '\n'.join([self.scopeformat.format(index=index, key=uppercase(key, withops=True), values=values) for index, key, values in zip(range(len(self.xarray.attrs)), self.xarray.attrs.keys(), self.xarray.attrs.values())])
    @property
    def variablestrings(self): return '\n'.join([self.variableformat.format(index=index, key=uppercase(key, withops=True), values=values.name()) for index, key, values in zip(range(len(self.variables)), self.variables.keys(), self.variables.values())])  

    def __add__(self, other): return self.add(other)
    def __sub__(self, other): return self.subtract(other)
    def __mul__(self, other): return self.multiply(other)
    def __truediv__(self, other): return self.divide(other)
    
    def __getitem__(self, items): 
        assert isinstance(items, dict)
        indexitems = {key:value for key, value in items.items() if isinstance(value, int)}
        keyitems = {key:value for key, value in items.items() if isinstance(value, str)}
        sliceitems = {key:value for key, value in items.items() if isinstance(value, slice)}
        assert len(indexitems) + len(sliceitems) + len(keyitems) == len(items)
        assert all([key in self.xarray.dims for key in keyitems.keys()])
        assert all([value in self.xarray.coords[key] for key, value in keyitems.items()])
        xarray = self.xarray[sliceitems]
        keyitems.update({key:list(self.xarray.coords.values())[index] for key, index in indexitems.items()})
        xarray = xarray[indexitems].loc[keyitems]
        xarray.attrs.update(keyitems)
        return self.__class__(data=xarray, key=self.key, name=self.name, variables=self.variables)
           
    def __getattr__(self, attr): 
        try: operation_function = OPERATIONS[attr]
        except KeyError: raise AttributeError('{}.{}'.format(self.__class__.__name__, attr))
        
        def wrapper(other, *args, **kwargs): return operation_function(self, other, *args, **kwargs)
        update_wrapper(wrapper, operation_function)   
        return wrapper

#    def flatten(self): 
#        dataframe = dataframe_fromxarray(self.xarray, self.key)
#        dataframe[self.key] = dataframe[self.key].apply(lambda x: str(self.variables[self.key](x)))
#        return FlatTable(data=dataframe, name=self.name, variables=self.variables)     


class FlatTable(TableBase):
    dataformat = 'DATA = {key}:\n{values}'
    variableformat = 'VARIABLE[{index}] = {key}: {values}' 

    def __init__(self, *args, variables, **kwargs):
        super().__init__(*args, **kwargs)
        self.__variables = variables.__class__({key:value for key, value in variables.items() if key in self.items()})
            
    @property
    def dataframe(self): return self.data
    @property
    def variables(self): return self.__variables

    @property
    def dim(self): return self.dataframe.ndim
    @property
    def shape(self): return self.dataframe.shape

    def items(self): return [column for column in self.dataframe.columns]
    def todict(self): return dict(data=self.dataframe, name=self.name, variables=self.variables)

    @property
    def strings(self): return '\n\n'.join([self.datastring, self.variablestrings])
    @property
    def datastring(self): return self.dataformat.format(key='DataFrame', values=str(self.dataframe))    
    @property
    def variablestrings(self): return '\n'.join([self.variableformat.format(index=index, key=uppercase(key, withops=True), values=values.name()) for index, key, values in zip(range(len(self.variables)), self.variables.keys(), self.variables.values())])  

    def __getitem__(self, key): 
        return self.__class__(data=self.dataframe[key], name=self.name, variables=self.variables)
    def __setitem__(self, key, items): 
        assert isinstance(items, dict)
        axes = _aslist(items.pop('axes'))
        if len(axes) == 1: newitems = self.createdata(key, fromcolumns='single',  axis=axes[0], **items)
        elif len(axes) > 1: newitems = self.createdata(key ,fromcolumns='multiple', axes=axes, **items)
        else: raise ValueError(axes)    
        self = self.__class__(**newitems, name=self.name)  
        
    @property
    def datacolumns(self): return [column for column in self.dataframe.columns if column not in self.scopecolumns]
    @property
    def scopecolumns(self): return [column for column in self.dataframe.columns if len(set(self.dataframe[column].values)) == 1]
    
    def unflatten(self, datakey, headerkeys, scopekeys):
        assert all([item in self.scopecolumns for item in _aslist(scopekeys)])
        headerkeys.sort(key=lambda key: len(set(self.dataframe[key].values)))
        dataframe = self.dataframe[[datakey, *_aslist(headerkeys), *_aslist(scopekeys)]]
        try: dataframe[datakey] = dataframe[datakey].apply(lambda x: self.variables[datakey].fromstr(x).value)
        except: pass
        return ArrayTable(data=xarray_fromdataframe(dataframe, datakey=datakey, headerkeys=headerkeys, scopekeys=scopekeys), key=datakey, name=self.name, variables=self.variables)
    
    @keydispatcher('fromcolumns')
    def createdata(self, key, *args, **kwargs): raise KeyError(key)
    
    @createdata.register('single')
    def __createdata_fromcolumn(self, key, *args, axis, function, variable_function, **kwargs):      
        dataframe, variables = self.dataframe, self.variables
        wrapper = lambda item: str(function(variables[axis].fromstr(item)))
        dataframe[key] = dataframe[axis].apply(wrapper)        
        variables[key] = variable_function(variables[axis])
        return dict(data=dataframe, variables=variables)

    @createdata.register('multiple')
    def __createdata_fromcolumns(self, key, *args, axes, function, variable_function, **kwargs):
        dataframe, variables = self.dataframe, self.variables
        wrapper = lambda items: str(function(*[variables[axis].fromstr(items[index]) for axis, index in zip(axes, range(len(axes)))]))
        dataframe[key] = dataframe[axes].apply(wrapper, axis=1)       
        variables[key] = variable_function(*[variables[axis] for axis in axes])
        return dict(data=dataframe, variables=variables)
  

class GeoTable(TableBase):
    dataformat = 'DATA = {key}:\n{values}'

    def __init__(self, *args, data, geodata, **kwargs):
        geodataframe = geodata.set_index('geography', drop=True)
        dataframe = data[['geoname', 'geoid', 'geography']].drop_duplicates().set_index('geography', drop=True)
        if 'geoid' not in geodataframe.columns: geodataframe['geoid'] = dataframe['geoid']
        if 'geoname' not in geodataframe.columns: geodataframe['geoname'] = dataframe['geoname']
        geodataframe = geodataframe.reset_index(drop=False)
        super().__init__(*args, data=geodataframe, **kwargs)        

    @property
    def dataframe(self): return self.data

    @property      
    def dim(self): return self.dataframe.ndim
    @property
    def shape(self): return self.dataframe.shape  

    @property
    def strings(self): return '\n\n'.join([self.datastring])
    @property
    def datastring(self): return self.dataformat.format(key='GeoDataFrame', values=str(self.dataframe)) 

    def todict(self): return dict(data=self.dataframe, name=self.name)
    def items(self): return [column for column in self.dataframe.columns]   
    
    
    
    
    
    
    
    
    
    
    
    
    