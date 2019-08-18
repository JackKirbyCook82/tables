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
from collections import namedtuple as ntuple

from utilities.dataframes import dataframe_fromxarray
from utilities.xarrays import xarray_fromdataframe
from utilities.strings import uppercase
from utilities.dispatchers import clskey_singledispatcher as keydispatcher

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['ArrayTable', 'FlatTable']
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
Structure = ntuple('Structure', 'layers dim shape fields')


class TableBase(ABC):
    variableformat = 'VARIABLE[{index}] = {key}: {name}' 
    
    def __init__(self, *args, data, name, variables, **kwargs): 
        self.__data = data
        self.__name = name
        self.__variables = variables.__class__({key:value for key, value in variables.items() if key in self.items()})
   
    @property
    def data(self): return self.__data   
    @property
    def variables(self): return self.__variables
    
    @property
    def tablename(self): return '{}: {}'.format(self.__class__.__name__.upper(), uppercase(self.__name, withops=True))   
    @property
    def name(self): return self.__name
         
    def todict(self): return dict(data=self.dataframe, variables=self.variables, name=self.name)
    
    @property
    def variablestrings(self): return '\n'.join([self.variableformat.format(index=index, key=uppercase(key, withops=True), name=values.name()) for index, key, values in zip(range(len(self.variables)), self.variables.keys(), self.variables.values())])      
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
    @property
    def structure(self): return Structure(self.layers, self.dim, self.shape, self.fields)

    @abstractmethod
    def items(self): pass


class FlatTable(TableBase):
    dataformat = 'DATA:\n{values}' 

    def __init__(self, *args, data, variables, **kwargs):
        try: dataframe = data.to_frame()
        except: dataframe = data
        for column in dataframe.columns: 
            try: dataframe[column] = dataframe[column].apply(lambda x: str(variables[column](x)))
            except: pass        
        super().__init__(*args, data=dataframe, variables=variables, **kwargs)

        
    @property
    def dataframe(self): return self.data
    @property
    def series(self): return self.data.squeeze()

    @property
    def layers(self): return 1
    @property
    def dim(self): return self.dataframe.ndim
    @property
    def shape(self): return self.dataframe.shape

    def items(self): return [column for column in self.dataframe.columns]

    @property
    def strings(self): return '\n\n'.join([self.datastring, self.variablestrings])
    @property
    def datastring(self): return self.dataformat.format(values=str(self.dataframe))    

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
    
    def unflatten(self, *datakeys, **kwargs):
        dataframe = self.dataframe.copy()
        for datakey in datakeys:
            try: dataframe.loc[:, datakey] = dataframe[datakey].apply(lambda x: self.variables[datakey].fromstr(x).value)
            except: pass        
        xarray = xarray_fromdataframe(dataframe, datakeys=datakeys, forcedataset=True, **kwargs)
        return ArrayTable(data=xarray, variables=self.variables, name=self.name)

    def todataframe(self, columns, index=None):
        if index: dataframe = self.dataframe.set_index(index, drop=True)
        else: dataframe = self.dataframe
        return dataframe[_aslist(columns)]
    
    def toseries(self, column, index=None):
        assert isinstance(column, str)
        if index: dataframe = self.dataframe.set_index(index, drop=True)
        else: dataframe = self.dataframe
        return dataframe[column].squeeze()


class ArrayTable(TableBase):
    dataformat = 'DATA[{index}] = {key}, {axes}:\n{values}'
    headerformat = 'HEADER[{index}] = {key}:\n{values}'
    scopeformat = 'SCOPE[{index}] = {key}: {values}'    
    labelformat = 'LABELS = {}'
    
    def __init__(self, *args, data, variables, **kwargs):
        assert isinstance(data, (xr.Dataset))
        super().__init__(*args, data=data, variables=variables, **kwargs)  

    @property
    def dataset(self): return self.data  
    @property
    def dataarrays(self): 
        dataarrays = {datakey:self.dataset[datakey] for datakey in self.datakeys}
        for datakey, dataarray in dataarrays.items(): dataarray.attrs = self.dataset.attrs
        return dataarrays
    @property
    def arrays(self): return {datakey:self.dataset[datakey].values for datakey in self.datakeys}
    
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
    
    def items(self): return [*self.datakeys, *self.headerkeys, *self.scopekeys]
    
    @property
    def datakeys(self): return tuple(self.dataset.data_vars.keys())
    @property
    def headerkeys(self): return tuple(self.dataset.dims)
    @property
    def scopekeys(self): return tuple(self.dataset.attrs.keys())
        
    @property
    def headers(self): return {key:tuple([value for value in self.dataset.coords[key].values]) for key in self.dataset.dims}
    @property
    def scope(self): return {key:value for key, value in self.dataset.attrs.items()}
    
    @property
    def strings(self): return '\n\n'.join([self.datastrings, self.headerstrings, self.scopestrings, self.variablestrings])
    @property
    def datastrings(self): 
        return '\n\n'.join([self.dataformat.format(index=index, key=uppercase(key, withops=True), values=self.dataset.data_vars[key].values, axes=tuple([uppercase(axes, withops=True) for axes in self.dataset[key].dims])) for index, key in zip(range(len(self.datakeys)), self.datakeys)])
    @property
    def headerstrings(self): 
        return '\n'.join([self.headerformat.format(index=index, key=uppercase(key, withops=True), values=self.dataset.coords[key].values) for index, key in zip(range(len(self.headerkeys)), self.headerkeys)])
    @property
    def scopestrings(self): 
        return '\n'.join([self.scopeformat.format(index=index, key=uppercase(key, withops=True), values=self.dataset.attrs[key]) for index, key in zip(range(len(self.scopekeys)), self.scopekeys)])

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

    def sortall(self, ascending=True):
        table = self
        for axis in self.headerkeys: table = table.sort(axis, ascending=ascending)
        return table

    def transpose(self, *axes):
        assert all([key in self.headerkeys for key in axes])
        order = axes + tuple([key for key in self.headerkeys if key not in axes])
        newdataset = self.dataset.transpose(*order)
        newdataset.attrs = self.dataset.attrs
        return self.__class__(data=newdataset, variables=self.variables, name=self.name)

    def expand(self, axis):
        newdataset = self.dataset.assign_coords(**{axis:self.dataset.attrs[axis]}).expand_dims(axis)        
        newdataset.attrs.pop(axis)
        return self.__class__(data=newdataset, variables=self.variables, name=self.name)
    
    def squeeze(self, axis):
        assert len(self.headers[axis]) == 1
        newdataset = self.dataset.squeeze(dim=axis, drop=True)
        newdataset.attrs = self.dataset.attrs
        newdataset.attrs.update({axis:self.headers[axis][0]})
        return self.__class__(data=newdataset, variables=self.variables, name=self.name)

    def __mul__(self, factor): return self.multiply(factor)
    def multiply(self, factor, *args, **kwargs):
        assert isinstance(factor, Number)
        if factor == 1: return self
        newdataset = self.dataset * factor
        newdataset.attrs = self.dataset.attrs
        newvariables = self.variables.update({datakey:self.variables[datakey].factor(*args, how='multiply', factor=factor, **kwargs) for datakey in _aslist(self.datakeys)})
        return self.__class__(data=newdataset, variables=newvariables, name=self.name)    
    
    def __truediv__(self, factor): return self.divide(factor)
    def divide(self, factor, *args, **kwargs):
        assert isinstance(factor, Number)
        if factor == 1: return self
        newdataset = self.dataset / factor
        newdataset.attrs = self.dataset.attrs
        newvariables = self.variables.update({datakey:self.variables[datakey].factor(*args, how='divide', factor=factor, **kwargs) for datakey in _aslist(self.datakeys)})
        return self.__class__(data=newdataset, variables=newvariables, name=self.name)    

    def flatten(self): 
        dataframe = dataframe_fromxarray(self.dataset) 
        for datakey in self.datakeys: dataframe[datakey] = dataframe[datakey].apply(lambda x: str(self.variables[datakey](x)))
        return FlatTable(data=dataframe, variables=self.variables, name=self.name)     




    
    
    
    
    
    
    
    
