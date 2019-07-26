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


_buffer = lambda : _OPTIONS['bufferchar'] * _OPTIONS['linewidth']
_aslist = lambda items: [items] if not isinstance(items, (list, tuple)) else list(items)


class TableBase(ABC):
    def __init__(self, *args, data, **kwargs): self.__data = data
   
    @property
    def data(self): return self.__data   
    @property
    def tablename(self): return self.__class__.__name__.upper()   
    
    def __str__(self): return '\n'.join([_buffer(), '\n\n'.join([self.tablename, self.strings, 'Layers={}, Dims={}, Shape={}'.format(self.layers, self.dim, self.shape)]), _buffer()])        
    def __len__(self): return self.dim    
    
    @abstractmethod
    def strings(self): pass
    @abstractmethod
    def layers(self): pass
    @abstractmethod
    def dim(self): pass
    @abstractmethod
    def shape(self): pass  

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
    def todict(self): return dict(data=self.dataframe, variables=self.variables)

    @property
    def strings(self): return '\n\n'.join([self.datastring, self.variablestrings])
    @property
    def datastring(self): return self.dataformat.format(values=str(self.dataframe))    
    @property
    def variablestrings(self): return '\n'.join([self.variableformat.format(index=index, key=uppercase(key, withops=True), name=values.name()) for index, key, values in zip(range(len(self.variables)), self.variables.keys(), self.variables.values())])  

    def __getitem__(self, key): 
        return self.__class__(data=self.dataframe[key], variables=self.variables)
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
        return dict(data=dataframe, variables=variables)

    @createdata.register('multiple')
    def __createdata_fromcolumns(self, key, *args, axes, function, **kwargs):
        dataframe, variables = self.dataframe, self.variables
        wrapper = lambda items: function(*[variables[axis].fromstr(items[index]) for axis, index in zip(axes, range(len(axes)))])
        dataframe[key] = dataframe[axes].apply(wrapper, axis=1)    
        variables[key] = type(dataframe[key].loc[0])
        dataframe[key] = dataframe[key].apply(str)
        return dict(data=dataframe, variables=variables)    
    
    def unflatten(self, datakeys, headerkeys, scopekeys):
        datakeys, headerkeys, scopekeys = [_aslist(item) for item in (datakeys, headerkeys, scopekeys)]
        assert all([len(set(self.dataframe[key].values)) == 1 for key in scopekeys if key in self.dataframe.columns])
        headerkeys.sort(key=lambda key: len(set(self.dataframe[key].values)))
        dataframe = self.dataframe[[*datakeys, *headerkeys, *scopekeys]]
        for datakey in datakeys:
            try: dataframe.loc[:, datakey] = dataframe[datakey].apply(lambda x: self.variables[datakey].fromstr(x).value)
            except: pass
        xarray = xarray_fromdataframe(dataframe, axekeys=headerkeys, scopekeys=scopekeys, forcedataset=True)
        return ArrayTable(data=xarray, variables=self.variables)


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
    def variables(self): return self.__variables        
    @property
    def dataset(self): return self.data  
    @property
    def dataarrays(self): 
        items = {key:self.dataset[key] for key in self.datakeys}
        for item in items.values(): item.attrs = self.dataset.attrs
        return items
    
    @property
    def layers(self): return len(self.dataset.data_vars)
    @property
    def dim(self): return len(self.dataset.dims)
    @property
    def shape(self): return tuple(self.dataset.sizes.values())
    
    @property
    def datakeys(self): return tuple(self.dataset.data_vars.keys())
    @property
    def headerkeys(self): return self.dataset.dims
    @property
    def scopekeys(self): return tuple(self.dataset.attrs.keys())
    
    def items(self): return [*self.datakey, *self.headerkeys, *self.scopekeys]
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
        elif isinstance(items, str):
            assert items in self.datakeys
            newdataset = self.dataset[items].to_dataset(name=items)
        elif isinstance(items, dict):
            indexitems = {key:value for key, value in items.items() if isinstance(value, int)}
            keyitems = {key:value for key, value in items.items() if isinstance(value, str)}
            sliceitems = {key:value for key, value in items.items() if isinstance(value, slice)}
            assert len(indexitems) + len(sliceitems) + len(keyitems) == len(items)
            assert all([key in self.headerkeys for key in keyitems.keys()])
            assert all([value in self.headervalues(key) for key, value in keyitems.items()])
            newdataset = self.dataset[sliceitems]
            keyitems.update({key:list(self.headervalues(index)) for key, index in indexitems.items()})
            newdataset = newdataset[indexitems].loc[keyitems]
            newdataset.attrs.update(keyitems)           
        else: raise TypeError(type(items))
        return self.__class__(data=newdataset, variables=self.variables)

    def sort(self, axis, ascending=True):
        newdataset = self.dataset
        newdataset.coords[axis] = pd.Index([self.variables[axis].fromstr(item) for item in newdataset.coords[axis].values])
        newdataset = newdataset.sortby(axis, ascending=ascending)
        newdataset.coords[axis] = pd.Index([str(item) for item in newdataset.coords[axis].values], name=axis)
        newdataset.attrs = self.dataset.attrs
        return self.__class__(data=newdataset, variables=self.variables)

    def flatten(self): 
        dataframe = dataframe_fromxarray(self.dataset) 
        for datakey in self.datakeys: dataframe[datakey] = dataframe[datakey].apply(lambda x: str(self.variables[datakey](x)))
        return FlatTable(data=dataframe, variables=self.variables)     
    
    #def toframe(self, index=[]):
    #    index = _aslist(index)
    #    columns = [item for item in self.headerkeys if item not in index]       
    #    dataframe = self.xarray.to_dataframe()[self.datakey].to_frame().reset_index()  
    #    dataframe = pd.pivot_table(dataframe, columns=columns ,index=index, values=self.datakey)       
    #    setattr(dataframe, 'scope', self.xarray.attrs)
    #    dataframe.name = self.datakey
    #    return dataframe
 

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
    def layer(self): return 1
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

    
    
    
    
    
    
    
    