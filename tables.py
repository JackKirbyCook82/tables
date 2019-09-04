# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 2019
@name:   Table Objects
@author: Jack Kirby Cook

"""

from abc import ABC, abstractmethod
import pandas as pd
import xarray as xr
from numbers import Number
from collections import OrderedDict as ODict

from utilities.dataframes import dataframe_fromxarray
from utilities.xarrays import xarray_fromdataframe
from utilities.dispatchers import clskey_singledispatcher as keydispatcher
from utilities.strings import uppercase

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['ArrayTable', 'FlatTable']
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""


_aslist = lambda items: [items] if not isinstance(items, (list, tuple)) else list(items)
_tableview = lambda table: '\n\n'.join([uppercase(table.name, withops=True), str(table.data), str(table.variables)])
 

class TableBase(ABC):
    View = None
    @classmethod
    def factory(cls, View): 
        cls.View = View
        return cls
    
    def __init__(self, *args, data, name, variables, **kwargs): 
        self.__data = data
        self.__name = name
        self.__variables = variables.__class__(**{key:value for key, value in variables.items() if key in self.keys})
        
    @property
    def name(self): return self.__name       
    @property
    def data(self): return self.__data   
    @property
    def variables(self): return self.__variables

    def __str__(self): 
        if self.View is not None: return str(self.View(self))
        else: return _tableview(self)
    def __len__(self): return self.dim  
    def __eq__(self, other):
        assert isinstance(self, type(other))
        return all([self.data == other.data, self.variables == other.variables])
    def __ne__(self, other): return not self.__eq__(other)

    @abstractmethod
    def layers(self): pass
    @abstractmethod
    def dims(self): pass
    @abstractmethod
    def shape(self): pass  
    
    def todict(self): return dict(data=self.data, variables=self.variables, name=self.name) 
    def rename(self, name): 
        self.__name = name
        return self
    
    @abstractmethod
    def keys(self): pass
    

class FlatTable(TableBase):
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
    def dims(self): return self.dataframe.ndim
    @property
    def shape(self): return self.dataframe.shape
    
    @property
    def keys(self): return tuple(self.dataframe.columns)

    def retag(self, **tags): 
        self.dataframe.rename(columns=tags, inplace=True)
        for oldkey, newkey in tags.items(): self.variables[newkey] = self.variables[oldkey]
        return self

    def __getitem__(self, columns): return self.select(*columns)
    def __setitem__(self, column, items): 
        assert isinstance(items, dict)
        axes = _aslist(items.pop('axes'))
        if len(axes) == 1: newitems = self.createdata(column, fromcolumns='single',  axis=axes[0], **items)
        elif len(axes) > 1: newitems = self.createdata(column ,fromcolumns='multiple', axes=axes, **items)
        else: raise ValueError(axes)    
        self = self.__class__(**newitems)  

    def select(self, *columns): return self.__class__(data=self.dataframe[_aslist(columns)], variables=self.variables, name=self.name)
    def drop(self, *columns): return self.select(*[column for column in self.dataframe.columns if column not in columns])
    
    @keydispatcher('fromcolumns')
    def createdata(self, key, *args, **kwargs): raise KeyError(key)
    
    @createdata.register('single')
    def __createdata_fromcolumn(self, key, *args, axis, function, **kwargs):      
        dataframe, variables = self.dataframe.copy(), self.variables.copy()
        wrapper = lambda item: function(variables[axis].fromstr(item))
        dataframe[key] = dataframe[axis].apply(wrapper)          
        variables[key] = type(dataframe[key].loc[0])
        dataframe[key] = dataframe[key].apply(str)
        return dict(data=dataframe, variables=variables, name=self.name)

    @createdata.register('multiple')
    def __createdata_fromcolumns(self, key, *args, axes, function, **kwargs):
        dataframe, variables = self.dataframe.copy(), self.variables.copy()
        wrapper = lambda items: function(*[variables[axis].fromstr(items[index]) for axis, index in zip(axes, range(len(axes)))])
        dataframe[key] = dataframe[axes].apply(wrapper, axis=1)    
        variables[key] = type(dataframe[key].loc[0])
        dataframe[key] = dataframe[key].apply(str)
        return dict(data=dataframe, variables=variables, name=self.name)    
    
    def todataframe(self, columns, index=None):
        dataframe = self.dataframe.copy()
        if index: dataframe = dataframe.set_index(index, drop=True)
        for column in columns: dataframe[column] = dataframe[column].apply(lambda x: self.variables[column].fromstr(x).value)
        return dataframe[_aslist(columns)]
    
    def toseries(self, column, index=None):
        assert isinstance(column, str)
        dataframe = self.dataframe.copy()
        if index: dataframe = dataframe.set_index(index, drop=True)
        dataframe[column] = dataframe[column].apply(lambda x: self.variables[column].fromstr(x).value)
        return dataframe[column].squeeze()    
    
    def unflatten(self, *datakeys, **kwargs):
        dataframe = self.dataframe.copy()
        for datakey in datakeys:
            try: dataframe.loc[:, datakey] = dataframe[datakey].apply(lambda x: self.variables[datakey].fromstr(x).value)
            except: pass        
        xarray = xarray_fromdataframe(dataframe, datakeys=datakeys, forcedataset=True, **kwargs)
        return ArrayTable(data=xarray, variables=self.variables.copy(), name=self.name)


class ArrayTable(TableBase):
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
    def layers(self): return len(self.dataset.data_vars)
    @property
    def dims(self): return len(self.dataset.dims)
    @property
    def shape(self): return tuple(self.dataset.sizes.values())
    
    @property
    def datakeys(self): return tuple(self.dataset.data_vars.keys())
    @property
    def dimkeys(self): return tuple(self.dataset.dims.keys())
    @property
    def headerkeys(self): return self.dimkeys
    @property
    def scopekeys(self): return tuple(set(self.dataset.coords.keys()) - set(self.dataset.dims.keys()))
    @property
    def axeskeys(self): return tuple(self.dataset.coords.keys())
    @property
    def keys(self): return self.datakeys + self.axeskeys

    @property
    def headers(self): return ODict([(key, self.dataset.coords[key].values) for key in self.dimkeys])
    @property
    def scope(self): return {key:self.dataset.coords[key].values for key in self.scopekeys}
    
    #def vheader(self, axis): return [self.variables[axis].fromstr(value) for value in self.headers[axis]]
    #def vscope(self, axis): return self.variables[axis].fromstr(str(self.scope[axis]))
    
    def retag(self, **tags): 
        newdataset = self.dataset.rename(name_dict=tags)
        variables = self.variables
        for oldkey, newkey in tags.items(): variables[newkey] = variables.pop(oldkey)
        return self.__class__(data=newdataset, variables=variables, name=self.name)

    def __getitem__(self, items): 
        if isinstance(items, int): return self[self.datakeys[items]]
        elif isinstance(items, str): return self[[items]]        
        elif isinstance(items, (list, tuple)):
            assert all([item in self.datakeys for item in items])
            newdataset = self.dataset[items]
        elif isinstance(items, dict):
            indexitems = {key:value for key, value in items.items() if isinstance(value, int)}
            keyitems = {key:value for key, value in items.items() if isinstance(value, str)}
            sliceitems = {key:value for key, value in items.items() if isinstance(value, slice)}
            assert len(indexitems) + len(sliceitems) + len(keyitems) == len(items) 
            keyitems.update({key:tuple(self.dataset.coords[key].values)[index] for key, index in indexitems.items()})
            assert all([key in tuple(self.dataset.dims) for key in keyitems.keys()])                                     
            newdataset = self.dataset[sliceitems]          
            newdataset = newdataset.loc[keyitems]                  
        else: raise TypeError(type(items))        
        newdataset.attrs = self.dataset.attrs
        table = self.__class__(data=newdataset, variables=self.variables.copy(), name=self.name)
        return table.dropallna()

    def sort(self, axis, ascending=True):
        assert axis in self.dimkeys
        newdataset = self.dataset.copy()
        newdataset.coords[axis] = pd.Index([self.variables[axis].fromstr(item) for item in newdataset.coords[axis].values])
        newdataset = newdataset.sortby(axis, ascending=ascending)
        newdataset.coords[axis] = pd.Index([str(item) for item in newdataset.coords[axis].values], name=axis)
        newdataset.attrs = self.dataset.attrs
        return self.__class__(data=newdataset, variables=self.variables.copy(), name=self.name)
    
    def sortall(self, ascending=True):
        table = self
        for axis in self.dimkeys: table = table.sort(axis, ascending=ascending)
        return table

    def dropna(self, axis):
        assert axis in self.dimkeys
        newdataset = self.dataset.copy()
        newdataset = newdataset.dropna(axis, how='all')
        return self.__class__(data=newdataset, variables=self.variables.copy(), name=self.name)

    def dropallna(self):
        table = self
        for axis in self.dimkeys: table = table.dropna(axis)
        return table

    def transpose(self, *axes):
        assert all([key in self.axeskeys for key in axes])
        order = axes + tuple([key for key in self.axes if key not in axes])
        newdataset = self.dataset.transpose(*order)
        newdataset.attrs = self.dataset.attrs
        return self.__class__(data=newdataset, variables=self.variables.copy(), name=self.name)

    def expand(self, axis):
        assert axis in self.scopekeys
        newdataset = self.dataset.expand_dims(axis)        
        newdataset.attrs = self.dataset.attrs
        return self.__class__(data=newdataset, variables=self.variables.copy(), name=self.name)
    
    def squeeze(self, *axes):
        assert all([len(self.dataset.coords[axis]) == 1 for axis in axes])
        newdataset = self.dataset
        for axis in axes: 
            if axis in self.scopekeys: pass
            elif axis in self.headerkeys: newdataset = newdataset.squeeze(dim=axis, drop=False)
            else: raise ValueError(axis)
        newdataset.attrs = self.dataset.attrs
        return self.__class__(data=newdataset, variables=self.variables.copy(), name=self.name)

    def removescope(self, *axes):
        assert all([axis in self.scopekeys for axis in axes])
        newdataset, newvariables = self.dataset, self.variables.copy()
        for axis in axes: 
            newdataset = newdataset.drop(axis)
            newvariables.pop(axis)
        newdataset.attrs = self.dataset.attrs        
        return self.__class__(data=newdataset, variables=newvariables, name=self.name)

    def __mul__(self, factor): return self.multiply(factor)
    def __truediv__(self, factor): return self.divide(factor)
    
    def multiply(self, factor, *args, **kwargs):
        assert isinstance(factor, Number)
        if factor == 1: return self
        newdataset = self.dataset * factor
        newdataset.attrs = self.dataset.attrs
        newvariables = self.variables.copy()
        newvariables.update({datakey:newvariables[datakey].factor(factor, *args, how='multiply', **kwargs) for datakey in _aslist(self.datakeys)})
        return self.__class__(data=newdataset, variables=newvariables, name=self.name)    
        
    def divide(self, factor, *args, **kwargs):
        assert isinstance(factor, Number)
        if factor == 1: return self
        newdataset = self.dataset / factor
        newdataset.attrs = self.dataset.attrs
        newvariables = self.variables.copy()        
        newvariables.update({datakey:newvariables[datakey].factor(factor, *args, how='divide', **kwargs) for datakey in _aslist(self.datakeys)})
        return self.__class__(data=newdataset, variables=newvariables, name=self.name)    

    def flatten(self): 
        dataset = self.dataset.copy()
        dataframe = dataframe_fromxarray(dataset) 
        for datakey in self.datakeys: dataframe[datakey] = dataframe[datakey].apply(lambda x: str(self.variables[datakey](x)))
        return FlatTable(data=dataframe, variables=self.variables.copy(), name=self.name)     




    
    
    
    
    
    
    
    
