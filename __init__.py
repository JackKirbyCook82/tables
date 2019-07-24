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


_OPTIONS = dict(linewidth=100, maxrows=30, maxcolumns=10, threshold=100, precision=3, bufferchar='=', fixednotation=True)
_PDMAPPING = {"display.max_rows":"maxrows", "display.max_columns":"maxcolumns"}
_NPMAPPING = {"linewidth":"linewidth", "threshold":"threshold", "precision":"precision", "suppress":"fixednotation"}

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
    
    def __init__(self, *args, datakey, variables, **kwargs):
        self.__datakey = datakey
        super().__init__(*args, **kwargs)
        self.__variables = variables.__class__([(key, variables[key]) for key in (self.datakey, *self.headerkeys, *self.scopekeys)])
           
    @property
    def xarray(self): return self.data  
    @property
    def narray(self): return self.data.values
    @property
    def variables(self): return self.__variables
    
    @property
    def dim(self): return len(self.xarray.dims)
    @property
    def shape(self): return self.xarray.shape
    
    @property
    def datakey(self): return self.__datakey
    @property
    def headerkeys(self): return self.xarray.dims
    def headervalues(self, key): 
        if isinstance(key, int): return self.headervalues(self.headerkeys[key])
        elif isinstance(key, str): return list(self.xarray.coords[key].values)
        else: raise TypeError(key)
    @property
    def headers(self): return {key:self.headervalues(key) for key in self.headerkeys}
    @property
    def scopekeys(self): return tuple(self.xarray.attrs.keys())
    def scopevalue(self, key): return self.xarray.attrs[key]
    @property
    def scope(self): return {key:self.scopevalue(key) for key in self.scopekeys}
        
    def axiskey(self, axis): return self.xarray.dim[axis] if isinstance(axis, int) else axis
    def axisindex(self, axis): return self.xarray.get_axis_num(axis) if isinstance(axis, str) else axis
    def items(self): return [self.datakey, *self.xarray.attrs.keys(), *self.xarray.coords.keys()]
    def todict(self): return dict(data=self.xarray, datakey=self.datakey, name=self.name, variables=self.variables)
    
    @property
    def strings(self): return '\n\n'.join([self.datastring, self.headerstrings, self.scopestrings, self.variablestrings])
    @property
    def datastring(self): return self.dataformat.format(key=uppercase(self.datakey, withops=True), values=self.xarray.values)
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
        return self.__class__(data=xarray, datakey=self.datakey, name=self.name, variables=self.variables)
           
    def __getattr__(self, attr): 
        try: operation_function = OPERATIONS[attr]
        except KeyError: raise AttributeError('{}.{}'.format(self.__class__.__name__, attr))
        
        def wrapper(other, *args, **kwargs): return operation_function(self, other, *args, **kwargs)
        update_wrapper(wrapper, operation_function)   
        return wrapper

    def sort(self, axis, ascending=True):
        xarray = self.xarray
        xarray.coords[axis] = pd.Index([self.variables[axis].fromstr(item) for item in xarray.coords[axis].values])
        xarray = xarray.sortby(axis, ascending=ascending)
        xarray.coords[axis] = pd.Index([str(item) for item in xarray.coords[axis].values], name=axis)
        return self.__class__(data=xarray, datakey=self.datakey, name=self.name, variables=self.variables)

    def flatten(self): 
        dataframe = dataframe_fromxarray(self.xarray)
        dataframe.rename(columns={list(set([column for column in dataframe.columns if column not in self.headerkeys]))[0]:self.datakey}, inplace=True)
        dataframe[self.datakey] = dataframe[self.datakey].apply(lambda x: str(self.variables[self.datakey](x)))
        return FlatTable(data=dataframe, name=self.name, variables=self.variables)     
    
    def toframe(self, index=[]):
        index = _aslist(index)
        columns = [item for item in self.headerkeys if item not in index]       
        dataframe = self.xarray.to_dataframe()[self.datakey].to_frame().reset_index()  
        dataframe = pd.pivot_table(dataframe, columns=columns ,index=index, values=self.datakey)       
        setattr(dataframe, 'scope', self.xarray.attrs)
        dataframe.name = self.datakey
        return dataframe


class FlatTable(TableBase):
    dataformat = 'DATA:\n{values}' 
    variableformat = 'VARIABLE[{index}] = {key}: {values}' 

    def __init__(self, *args, data, variables, **kwargs):
        try: dataframe = data.to_frame()
        except: dataframe = data
        for column in dataframe.columns: 
            try: dataframe[column] = dataframe[column].apply(lambda x: str(variables[column](x)))
            except: pass
        super().__init__(*args, data=dataframe, **kwargs)
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
    def datastring(self): return self.dataformat.format(values=str(self.dataframe))    
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

    @keydispatcher('fromcolumns')
    def createdata(self, key, *args, **kwargs): raise KeyError(key)
    
    @createdata.register('single')
    def __createdata_fromcolumn(self, key, *args, axis, function, **kwargs):      
        dataframe, variables = self.dataframe, self.variables
        wrapper = lambda item: function(variables[axis].fromstr(item))
        dataframe[key] = dataframe[axis].apply(wrapper)          
        variables[key] = type(dataframe[key].loc[0])
        dataframe[key] = dataframe[key].apply(str)
        return dict(data=dataframe, variables=variables)

    @createdata.register('multiple')
    def __createdata_fromcolumns(self, key, *args, axes, function, **kwargs):
        dataframe, variables = self.dataframe, self.variables
        wrapper = lambda items: function(*[variables[axis].fromstr(items[index]) for axis, index in zip(axes, range(len(axes)))])
        dataframe[key] = dataframe[axes].apply(wrapper, axis=1)    
        variables[key] = type(dataframe[key].loc[0])
        dataframe[key] = dataframe[key].apply(str)
        return dict(data=dataframe, variables=variables)    
    
    def unflatten(self, datakey, headerkeys, scopekeys):
        headerkeys, scopekeys = [_aslist(item) for item in (headerkeys, scopekeys)]
        assert all([len(set(self.dataframe[key].values)) == 1 for key in scopekeys if key in self.dataframe.columns])
        headerkeys.sort(key=lambda key: len(set(self.dataframe[key].values)))
        dataframe = self.dataframe[[datakey, *headerkeys, *scopekeys]]
        try: dataframe.loc[:, datakey] = dataframe[datakey].apply(lambda x: self.variables[datakey].fromstr(x).value)
        except: pass
        xarray = xarray_fromdataframe(dataframe, datakey=datakey, headerkeys=headerkeys, scopekeys=scopekeys)
        return ArrayTable(data=xarray, datakey=datakey, name=self.name, variables=self.variables)
    

class GeoTable(TableBase):
    dataformat = 'DATA:\n{values}'

    def __init__(self, *args, geodata, **kwargs):
        assert all([item in geodata.columns for item in ('geography', 'geometry')])   
        geodataframe = geodata.set_index('geography', drop=True)  
        super().__init__(*args, data=geodataframe, **kwargs)   

    @property
    def geodataframe(self): return self.data
    
    @property
    def index(self): return self.geodataframe.index.values
    
    @property      
    def dim(self): return self.geodataframe.ndim
    @property
    def shape(self): return self.geodataframe.shape  

    @property
    def strings(self): return '\n\n'.join([self.datastring])
    @property
    def datastring(self): return self.dataformat.format(values=str(self.geodataframe))  

    def todict(self): return dict(data=self.geodataframe, name=self.name)
    def items(self): return [column for column in self.geodataframe.columns]   
    

    
    
    
    
    
    
    
    