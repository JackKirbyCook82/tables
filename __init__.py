# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 2019
@name:   Table Objects
@author: Jack Kirby Cook

"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import xarray as xr
from numbers import Number

from utilities.dataframes import dataframe_fromxarray
from utilities.xarrays import xarray_fromdataframe
from utilities.strings import uppercase
from utilities.dispatchers import clskey_singledispatcher as keydispatcher

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


_ALL = '*'
_buffer = lambda : _OPTIONS['bufferchar'] * _OPTIONS['linewidth']
_aslist = lambda items: [items] if not isinstance(items, (list, tuple)) else list(items)


class TableBase(ABC):
    def __init__(self, *args, data, name, **kwargs): 
        self.__data = data
        self.__name = name
   
    @property
    def data(self): return self.__data   
    @property
    def tablename(self): return '{}: {}'.format(self.__class__.__name__.upper(), self.__name)   
    @property
    def name(self): return self.__name
    
    def __str__(self): return '\n'.join([_buffer(), '\n\n'.join([self.tablename, self.strings, 'Layers={}, Dims={}, Shape={}, Fields={}'.format(self.layers, self.dim, self.shape, self.fields)]), _buffer()])        
    def __len__(self): return self.dim    
    
    @abstractmethod
    def strings(self): pass
    @abstractmethod
    def layers(self): pass
    @abstractmethod
    def dim(self): pass
    @abstractmethod
    def shape(self): pass  
    @property
    def fields(self): return np.prod(self.shape)

    @abstractmethod
    def items(self): pass
    @abstractmethod
    def todict(self): pass

 
class FlatTable(TableBase):
    dataformat = 'DATA:\n{values}' 
    variableformat = 'VARIABLE[{index}] = {key}: {name}' 

    def __init__(self, *args, data, variables, **kwargs):
        try: dataframe = data.to_frame()
        except: dataframe = data
        for column in dataframe.columns: 
            try: dataframe[column] = dataframe[column].apply(lambda x: str(variables[column](x)))
            except: pass
        super().__init__(*args, data=dataframe, **kwargs)
        self.__variables = variables.__class__({key:value for key, value in variables.items() if key in self.items()})
                   
    @property
    def variables(self): return self.__variables        
    @property
    def dataframe(self): return self.data

    @property
    def layers(self): return 1
    @property
    def dim(self): return self.dataframe.ndim
    @property
    def shape(self): return self.dataframe.shape

    def items(self): return [column for column in self.dataframe.columns]
    def todict(self): return dict(data=self.dataframe, variables=self.variables, name=self.name)

    @property
    def strings(self): return '\n\n'.join([self.datastring, self.variablestrings])
    @property
    def datastring(self): return self.dataformat.format(values=str(self.dataframe))    
    @property
    def variablestrings(self): return '\n'.join([self.variableformat.format(index=index, key=uppercase(key, withops=True), name=values.name()) for index, key, values in zip(range(len(self.variables)), self.variables.keys(), self.variables.values())])  

    def __getitem__(self, key): 
        return self.__class__(data=self.dataframe[key], variables=self.variables, name=self.name)
    def __setitem__(self, key, items): 
        assert isinstance(items, dict)
        axes = _aslist(items.pop('axes'))
        if len(axes) == 1: newitems = self.createdata(key, fromcolumns='single',  axis=axes[0], **items)
        elif len(axes) > 1: newitems = self.createdata(key ,fromcolumns='multiple', axes=axes, **items)
        else: raise ValueError(axes)    
        self = self.__class__(**newitems)  

    @keydispatcher('fromcolumns')
    def createdata(self, key, *args, **kwargs): raise KeyError(key)
    
    @createdata.register('single')
    def __createdata_fromcolumn(self, key, *args, axis, function, **kwargs):      
        dataframe, variables = self.dataframe, self.variables
        wrapper = lambda item: function(variables[axis].fromstr(item))
        dataframe[key] = dataframe[axis].apply(wrapper)          
        variables[key] = type(dataframe[key].loc[0])
        dataframe[key] = dataframe[key].apply(str)
        return dict(data=dataframe, variables=variables, name=self.name)

    @createdata.register('multiple')
    def __createdata_fromcolumns(self, key, *args, axes, function, **kwargs):
        dataframe, variables = self.dataframe, self.variables
        wrapper = lambda items: function(*[variables[axis].fromstr(items[index]) for axis, index in zip(axes, range(len(axes)))])
        dataframe[key] = dataframe[axes].apply(wrapper, axis=1)    
        variables[key] = type(dataframe[key].loc[0])
        dataframe[key] = dataframe[key].apply(str)
        return dict(data=dataframe, variables=variables, name=self.name)    
    
    def unflatten(self, datakeys, headerkeys, scopekeys, *args, **kwargs):
        assert all([isinstance(item, (str, tuple, list)) for item in (datakeys, headerkeys, scopekeys)])
        datakeys, headerkeys, scopekeys = [_aslist(item) for item in (datakeys, headerkeys, scopekeys)]
                
        assert all([key in self.dataframe.columns for key in (*datakeys, *headerkeys, *scopekeys)])        
        assert all([len(set(self.dataframe[key].values)) == 1 for key in scopekeys if key in self.dataframe.columns])
        
        headerkeys.sort(key=lambda key: len(set(self.dataframe[key].values)))
        dataframe = self.dataframe[[*datakeys, *headerkeys, *scopekeys]]
        for datakey in datakeys:
            try: dataframe.loc[:, datakey] = dataframe[datakey].apply(lambda x: self.variables[datakey].fromstr(x).value)
            except: pass
        xarray = xarray_fromdataframe(dataframe, *args, datakeys=datakeys, axekeys=headerkeys, attrkeys=scopekeys, forcedataset=True, **kwargs)
        variables = {key:value for key, value in self.variables.items() if key in dataframe.columns}
        return ArrayTable(data=xarray, variables=variables, name=self.name)


class ArrayTable(TableBase):
    dataformat = 'DATA[{index}] = {key}:\n{values}'
    headerformat = 'HEADER[{index}] = {key}:\n{values}'
    scopeformat = 'SCOPE[{index}] = {key}: {values}'    
    variableformat = 'VARIABLE[{index}] = {key}: {name}' 
    
    def __init__(self, *args, data, variables, **kwargs):
        assert isinstance(data, (xr.Dataset))
        super().__init__(*args, data=data, **kwargs)  
        self.__variables = variables.__class__([(key, variables[key]) for key in (*self.datakeys, *self.headerkeys, *self.scopekeys)])
    
    @property
    def dataset(self): return self.data  
    @property
    def dataarrays(self): 
        items = {key:self.dataset[key] for key in self.datakeys}
        for item in items.values(): item.attrs = self.dataset.attrs
        return items
    
    @property
    def variables(self): return self.__variables 
    @property
    def datavariables(self): return {key:self.variables[key] for key in self.datakeys}        
    @property
    def axesvariables(self): return {key:self.variables[key] for key in self.axeskeys}
    @property
    def headervariables(self): return {key:self.variables[key] for key in self.headerkeys}
    @property
    def scopevariables(self): return {key:self.variables[key] for key in self.scopekeys}    
    
    @property
    def layers(self): return len(self.dataset.data_vars)
    @property
    def dim(self): return len(self.dataset.dims)
    @property
    def shape(self): return tuple(self.dataset.sizes.values())
    
    @property
    def datakeys(self): return tuple(self.dataset.data_vars.keys())
    @property
    def headerkeys(self): return tuple(self.dataset.dims)
    @property
    def scopekeys(self): return tuple(self.dataset.attrs.keys())
    @property
    def axeskeys(self): return (*self.headerkeys, *self.scopekeys)
    
    @property
    def headers(self): return {key:value.values for key, value in self.dataset.coords.items()}
    @property
    def scope(self): return {key:value for key, value in self.dataset.attrs.items()}
    @property
    def axes(self): return {key:(self.headers[key] if key in self.headerkeys else self.scope[key]) for key in self.axeskeys}
       
    def items(self): return [*self.datakeys, *self.headerkeys, *self.scopekeys]
    def todict(self): return dict(data=self.dataset, name=self.name, variables=self.variables)
    
    @property
    def strings(self): return '\n\n'.join([self.datastrings, self.headerstrings, self.scopestrings, self.variablestrings])
    @property
    def datastrings(self): return '\n\n'.join([self.dataformat.format(index=index, key=uppercase(key, withops=True), values=self.dataset.data_vars[key].values) for index, key in zip(range(len(self.datakeys)), self.datakeys)])
    @property
    def headerstrings(self): return '\n'.join([self.headerformat.format(index=index, key=uppercase(axis, withops=True), values=self.dataset.coords[axis].values) for index, axis in zip(range(len(self.headerkeys)), self.headerkeys)])
    @property
    def scopestrings(self): return '\n'.join([self.scopeformat.format(index=index, key=uppercase(key, withops=True), values=self.dataset.attrs[key]) for index, key in zip(range(len(self.scopekeys)), self.scopekeys)])
    @property
    def variablestrings(self): return '\n'.join([self.variableformat.format(index=index, key=uppercase(key, withops=True), name=values.name()) for index, key, values in zip(range(len(self.variables)), self.variables.keys(), self.variables.values())])  
    
    def __getitem__(self, items): 
        if isinstance(items, (list, tuple)):
            assert all([item in self.datakeys for item in items])
            newdataset = self.dataset[items]
            newdataset.attrs = self.dataset.attrs
        elif isinstance(items, int):
            return self[self.datakeys[items]]
        elif isinstance(items, str):
            assert items in self.datakeys
            newdataset = self.dataset[items].to_dataset(name=items)
            newdataset.attrs = self.dataset.attrs
        elif isinstance(items, dict):
            indexitems = {key:value for key, value in items.items() if isinstance(value, int)}
            keyitems = {key:value for key, value in items.items() if isinstance(value, str)}
            sliceitems = {key:value for key, value in items.items() if isinstance(value, slice)}
            assert len(indexitems) + len(sliceitems) + len(keyitems) == len(items)
 
            keyitems.update({key:self.headers[key][index] for key, index in indexitems.items()})
            assert all([key in self.headerkeys for key in keyitems.keys()])             
                        
            newdataset = self.dataset[sliceitems]          
            newdataset = newdataset.loc[keyitems]
            newdataset.attrs = self.dataset.attrs 
            for key, value in keyitems.items(): newdataset = newdataset.drop(key)
            newdataset.attrs.update(keyitems)                   
        else: raise TypeError(type(items))
        return self.__class__(data=newdataset, variables=self.variables, name=self.name)

    def update(self, **kwargs):
        for axis, values in kwargs.items():
            if axis in self.headerkeys:
                newdataset = self.dataset.assign_coords(**{axis:[str(value) for value in values]})
                newdataset.attrs = self.dataset.attrs
            elif axis in self.scopekeys:
                newdataset = self.dataset
                newdataset.attrs[axis] = str(values)
            else: raise ValueError(axis)        
        return self.__class__(data=newdataset, variables=self.variables, name=self.name)

    def sort(self, axis, ascending=True):
        newdataset = self.dataset
        newdataset.coords[axis] = pd.Index([self.variables[axis].fromstr(item) for item in newdataset.coords[axis].values])
        newdataset = newdataset.sortby(axis, ascending=ascending)
        newdataset.coords[axis] = pd.Index([str(item) for item in newdataset.coords[axis].values], name=axis)
        newdataset.attrs = self.dataset.attrs
        return self.__class__(data=newdataset, variables=self.variables, name=self.name)

    def expand(self, onscope):
        newdataset = self.dataset.assign_coords(**{onscope:self.dataset.attrs[onscope]}).expand_dims(onscope)        
        newdataset.attrs.pop(onscope)
        return self.__class__(data=newdataset, variables=self.variables, name=self.name)
    
    def squeeze(self, onaxis):
        assert len(self.headers[onaxis]) == 1
        newdataset = self.dataset.squeeze(dim=onaxis)
        newdataset.attrs = self.dataset.attrs
        newdataset.attrs.update({onaxis:self.headers[onaxis][0]})
        return self.__class__(data=newdataset, variables=self.variables, name=self.name)

    def multiply(self, factor, *args, datakeys, **kwargs):
        assert isinstance(factor, Number)
        if factor == 1: return self
        newdataarrays = [self.dataarrays[datakey] * factor if datakey in _aslist(datakeys) else self.dataarrays[datakey] for datakey in self.datakeys]
        newdataset = xr.merge(newdataarrays)  
        newdataset.attrs = self.dataset.attrs
        newvariables = self.variables.copy()
        newvariables.update({datakey:self.variables[datakey].factor(*args, how='multiply', factor=factor, **kwargs) for datakey in _aslist(datakeys)})
        return self.__class__(data=newdataset, variables=newvariables, name=self.name)    
    
    def divide(self, factor, *args, datakeys, **kwargs):
        assert isinstance(factor, Number)
        if factor == 1: return self
        newdataarrays = [self.dataarrays[datakey] / factor if datakey in _aslist(datakeys) else self.dataarrays[datakey] for datakey in self.datakeys]
        newdataset = xr.merge(newdataarrays) 
        newdataset.attrs = self.dataset.attrs
        newvariables = self.variables.copy()
        newvariables.update({datakey:self.variables[datakey].factor(*args, how='divide', factor=factor, **kwargs) for datakey in _aslist(datakeys)})
        return self.__class__(data=newdataset, variables=newvariables, name=self.name)      

    def flatten(self): 
        dataframe = dataframe_fromxarray(self.dataset) 
        for datakey in self.datakeys: dataframe[datakey] = dataframe[datakey].apply(lambda x: str(self.variables[datakey](x)))
        return FlatTable(data=dataframe, variables=self.variables, name=self.name)     


class GeoTable(TableBase):
    dataformat = 'DATA:\n{values}'

    def __init__(self, *args, data, **kwargs):
        assert all([item in data.columns for item in ('geography', 'geometry')])   
        geodataframe = data.set_index('geography', drop=True)  
        super().__init__(*args, data=geodataframe, **kwargs)   

    @property
    def geodataframe(self): return self.data
    
    @property
    def index(self): return self.geodataframe.index.values
    
    @property
    def layers(self): return 1
    @property      
    def dim(self): return self.geodataframe.ndim
    @property
    def shape(self): return self.geodataframe.shape  

    @property
    def strings(self): return '\n\n'.join([self.datastring])
    @property
    def datastring(self): return self.dataformat.format(values=str(self.geodataframe))  

    def items(self): return [column for column in self.geodataframe.columns]   
    def todict(self): return dict(data=self.geodataframe)

    
    
    
    
    
    
    
    
