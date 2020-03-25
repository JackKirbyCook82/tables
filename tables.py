# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 2019
@name:   Table Objects
@author: Jack Kirby Cook

"""

import os.path
from abc import ABC, abstractmethod
import pandas as pd
import openpyxl
import numpy as np
import xarray as xr
from scipy import stats
from numbers import Number
from collections import OrderedDict as ODict
from collections import namedtuple as ntuple

from utilities.dataframes import dataframe_fromxarray
from utilities.xarrays import xarray_fromdataframe, standardize, absolute
from utilities.dispatchers import clskey_singledispatcher as keydispatcher
from utilities.strings import uppercase

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['ArrayTable', 'FlatTable', 'HistTable']
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""


AGGREGATIONS = {'sum':np.sum, 'avg':np.mean, 'max':np.max, 'min':np.min}

_normalize = lambda x: x / np.sum(x)
_union = lambda x, y: list(set(x) | set(y))
_intersection = lambda x, y: list(set(x) & set(y))
_aslist = lambda items: [items] if not isinstance(items, (list, tuple)) else list(items)
_filterempty = lambda items: [item for item in _aslist(items) if item]

_replacenan = lambda dataarray, value: xr.where(~np.isnan(dataarray), dataarray, value)
_replaceinf = lambda dataarray, value: xr.where(~np.isinf(dataarray), dataarray, value)
_replaceneg = lambda dataarray, value: xr.where(dataarray >= 0, dataarray, value)
_replacestd = lambda dataarray, value, std: xr.where(absolute(standardize(dataarray)) < std, dataarray, value)


class HistArray(ntuple('HistArray', 'weightskey weights axiskey axis index scope')):
    def __new__(cls, weightskey, weights, axiskey, axis, index, scope):
        assert len(weights) == len(axis) == len(index)
        assert isinstance(scope, dict)
        return super().__new__(cls, weightskey, weights, axiskey, axis, index, scope)


class TableBase(ABC):
    View = lambda table: None
    @classmethod
    def factory(cls, view=None): 
        if view: cls.View = view
        return cls

    def __init__(self, data, *args, name, variables, **kwargs): 
        self.__data, self.__name = data, name
        self.__variables = variables.select([key for key in self.keys if key in variables.keys()])
     
    @property
    def name(self): return self.__name       
    @property
    def data(self): return self.__data   
    @property
    def variables(self): return self.__variables

    @property
    def view(self): return self.View(self)       
    def __str__(self):
        view = self.View(self)
        if view: return str(view)
        else: return '\n\n'.join([uppercase(self.name, withops=True), str(self.data), str(self.variables)])
    
    def __ne__(self, other): return not self.__eq__(other)
    def __eq__(self, other):
        assert isinstance(self, type(other))
        return all([self.data == other.data, self.variables == other.variables])
    
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
 

class HistTable(TableBase):
    def __call__(self, size): return self.__histogram.rvs(size=(1, size))[0]            
    def __init__(self, histarray, *args, variables, **kwargs):
        assert isinstance(histarray, HistArray)        
        super().__init__(histarray, *args, variables=variables, **kwargs)       
        self.__histogram = stats.rv_discrete(name=self.name, values=(self.index, _normalize(self.weights)))
     
    @property
    def concepts(self): return {indexvalue:axisvalue for indexvalue, axisvalue in zip(self.index, self.axis)}        
    @property
    def histarray(self): return self.data
    
    @property
    def weights(self): return self.data.weights
    @property
    def axis(self): return self.data.axis
    @property
    def index(self): return self.data.index    
    @property
    def scope(self): return self.data.scope
    @property
    def histogram(self): return self.__histogram
    
    @property
    def weightskey(self): return self.data.weightskey
    @property
    def axiskey(self): return self.data.axiskey
    @property
    def scopekeys(self): return tuple(self.data.scope.keys())
    @property
    def keys(self): return (self.data.weightskey, self.data.axiskey, *self.data.scope.keys())

    @property
    def layers(self): return 1
    @property
    def dims(self): return 2
    @property
    def shape(self): return (2, len(self.data.weights))
    
    def __len__(self): return len(self.data.weights)       
    def __iter__(self): 
        for axis, index, weight in zip(self.axis, self.index, self.weights):
            yield axis, index, weight
            
    def retag(self, **tags): 
        data = HistArray()
        variables, scope = self.variables.copy(), self.scope.copy()
        for oldkey, newkey in tags.items(): variables[newkey], scope[newkey] = variables[oldkey], scope[oldkey]
        data = HistArray(tags.get(self.data.weightskey, self.data.weightskey), self.data.weights, tags.get(self.data.axiskey, self.data.axiskey), self.data.axis, scope)
        return self.__class__(data, variables=variables, name=self.name)

    @property
    def array(self): return np.array([np.full(weight, index) for index, weight in zip(self.index, self.weights)]).flatten()        
    def total(self): return np.sum(self.weights)
    def mean(self): return self.histogram.mean()
    def median(self): return self.histogram.median()
    def std(self): return self.histogram.std()
    def rstd(self): return self.std() / self.mean()
    def skew(self): return stats.skew(self.array)
    def kurtosis(self): return stats.kurtosis(self.array)    


class FlatTable(TableBase):
    def __init__(self, data, *args, variables, **kwargs):
        try: dataframe = data.to_frame()
        except: dataframe = data
        for column in dataframe.columns: 
            try: dataframe[:, column] = dataframe[column].apply(lambda x: str(variables[column](x)))
            except: pass      
        super().__init__(dataframe, *args, variables=variables, **kwargs)
       
    @property
    def dataframe(self): return self.data
    @property
    def series(self): return self.data.squeeze()
    @property
    def columns(self): return self.data.columns
    @property
    def keys(self): return tuple(self.dataframe.columns)

    @property
    def layers(self): return 1
    @property
    def dims(self): return self.dataframe.ndim
    @property
    def shape(self): return self.dataframe.shape
    def __len__(self): return self.dims 
    
    def retag(self, **tags): 
        dataframe = self.dataframe.rename(columns=tags, inplace=True)
        variables = self.variables.copy()
        for oldkey, newkey in tags.items(): variables[newkey] = variables[oldkey]
        return self.__class__(dataframe, variables=variables, name=self.name)

    def __getitem__(self, columns): return self.select(*_filterempty(columns))
    def __setitem__(self, column, items): 
        assert isinstance(items, dict)
        axes = _aslist(items.pop('axes'))
        if len(axes) == 1: newitems = self.createdata('single', column, axis=axes[0], **items)
        elif len(axes) > 1: newitems = self.createdata('multiple', column, axes=axes, **items)
        else: raise ValueError(axes)  
        data = newitems.pop('data')
        self = self.__class__(data, **newitems)  

    def select(self, *columns): return self.__class__(self.dataframe[_aslist(columns)], variables=self.variables, name=self.name)
    def drop(self, *columns): return self.select(*[column for column in self.dataframe.columns if column not in columns])
    
    @keydispatcher
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
   
    def pivot(self, index=[], columns=[], values=[], aggs={}):
        index, columns, values = [_aslist(item) for item in (index, columns, values)]
        dataframe = self.dataframe.copy()
        for item in index: dataframe[item] = dataframe[item].apply(lambda x: self.variables[item].fromstr(x))
        for value in values: dataframe[value] = dataframe[value].apply(lambda x: self.variables[value].fromstr(x).value)
        aggs = {value:AGGREGATIONS[aggkey] for value, aggkey in aggs.items() if value in values}
        pivoted = dataframe.pivot_table(index=index, columns=columns, values=values, aggfunc=aggs)
        pivoted = pivoted.sort_index(ascending=True)
        return pivoted

    def toseries(self, value, index=[]):
        dataframe = self.dataframe[[value, *_aslist(index)]]
        if index: dataframe = dataframe.set_index(index, drop=True)
        dataframe.loc[:, value] = dataframe[value].apply(lambda x: self.variables[value].fromstr(x).value)
        return dataframe.squeeze()
        
    def todataframe(self, *values, index=[]):
        dataframe = self.dataframe[[*values, *_aslist(index)]]
        if index: dataframe = dataframe.set_index(index, drop=True)
        for value in values: dataframe.loc[:, value] = dataframe[value].apply(lambda x: self.variables[value].fromstr(x).value)
        return dataframe
    
    def unflatten(self, *datakeys, **kwargs):
        dataframe = self.dataframe.copy(deep=True)
        for datakey in datakeys: dataframe.loc[:, datakey] = dataframe[datakey].apply(lambda x: self.variables[datakey].fromstr(x).value)      
        xarray = xarray_fromdataframe(dataframe, datakeys=datakeys, forcedataset=True, **kwargs)
        return ArrayTable(xarray, variables=self.variables.copy(), name=self.name)

    def toexcel(self, file):
        if not os.path.isfile(file): openpyxl.Workbook().save(file)            
        writer = pd.ExcelWriter(file, engine='openpyxl') 
        writer.book = openpyxl.load_workbook(file)
        self.dataframe.to_excel(writer, sheet_name=self.name, index=False, header=True, engine='openpyxl') 
        writer.save()
        writer.close()


class ArrayTable(TableBase):
    def __init__(self, data, *args, variables, **kwargs):
        assert isinstance(data, (xr.Dataset))
        super().__init__(data, *args, variables=variables, **kwargs)  

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
    def spans(self): return {datakey:(np.nanmin(self.dataarrays[datakey].values), np.nanmax(self.dataarrays[datakey].values)) for datakey in self.datakeys} 
    @property
    def mins(self): return {datakey:np.nanmin(self.dataarrays[datakey].values) for datakey in self.datakeys}
    @property
    def maxs(self): return {datakey:np.nanmax(self.dataarrays[datakey].values) for datakey in self.datakeys}
    
    @property
    def layers(self): return len(self.dataset.data_vars)
    @property
    def dims(self): return len(self.dataset.dims)
    @property
    def shape(self): return tuple(self.dataset.sizes.values())
    def __len__(self): return self.dims 
    
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
    
    def retag(self, **tags): 
        newdataset = self.dataset.rename(name_dict=tags)
        variables = self.variables.copy()
        for oldkey, newkey in tags.items(): variables[newkey] = variables.pop(oldkey)
        return self.__class__(newdataset, variables=variables, name=self.name)

    def __getitem__(self, items):
        if not items: return self
        elif isinstance(items, int): return self[self.datakeys[items]]
        elif isinstance(items, str): return self[[items]]        
        elif isinstance(items, (list, tuple)):
            items = _filterempty(items)
            assert all([item in self.datakeys for item in items])
            newdataset = self.dataset[items]
        elif isinstance(items, dict):
            try: return self.sel(**items)
            except: return self.isel(**items)
        else: raise TypeError(type(items))        
        newdataset.attrs = self.dataset.attrs
        return self.__class__(newdataset, variables=self.variables.copy(), name=self.name).dropallna()
     
    def isel(self, **axis):
        assert all([key in self.headerkeys for key in axis.keys()])
        assert all([isinstance(value, (int, slice, list)) for value in axis.values()])
        for value in axis.values(): 
            if isinstance(value, list): assert all([isinstance(item, int) for item in value])
        newdataset = self.dataset.isel(**axis)
        newdataset.attrs = self.dataset.attrs
        return self.__class__(newdataset, variables=self.variables.copy(), name=self.name).dropallna()          
            
    def sel(self, **axis):
        assert all([key in self.headerkeys for key in axis.keys()])
        assert all([isinstance(value, (str, list)) for value in axis.values()])
        for value in axis.values(): 
            if isinstance(value, list): assert all([isinstance(item, str) for item in value])
        newdataset = self.dataset.sel(**axis)
        newdataset.attrs = self.dataset.attrs
        return self.__class__(newdataset, variables=self.variables.copy(), name=self.name).dropallna()  
    
    def sort(self, axis, ascending=True):
        assert axis in self.dimkeys
        newdataset = self.dataset.copy()
        newdataset.coords[axis] = pd.Index([self.variables[axis].fromstr(item) for item in newdataset.coords[axis].values])
        newdataset = newdataset.sortby(axis, ascending=ascending)
        newdataset.coords[axis] = pd.Index([str(item) for item in newdataset.coords[axis].values], name=axis)
        newdataset.attrs = self.dataset.attrs
        return self.__class__(newdataset, variables=self.variables.copy(), name=self.name)
    
    def sortall(self, ascending=True):
        table = self
        for axis in self.dimkeys: table = table.sort(axis, ascending=ascending)
        return table

    def dropna(self, axis):
        assert axis in self.dimkeys
        newdataset = self.dataset.copy()
        newdataset = newdataset.dropna(axis, how='all')
        return self.__class__(newdataset, variables=self.variables.copy(), name=self.name)

    def dropallna(self):
        table = self
        for axis in self.dimkeys: table = table.dropna(axis)
        return table

    def fillna(self, fill=None, **fillvalues):
        newdataarrays = {datakey:_replacenan(self.dataarrays[datakey], fillvalues.get(datakey, fill)) for datakey, dataarray in self.dataarrays.items()}
        newdataset = xr.merge([value.to_dataset(name=key) for key, value in newdataarrays.items()], join='outer') 
        newdataset.attrs = self.dataset.attrs
        return self.__class__(newdataset, variables=self.variables.copy(), name=self.name)
    
    def fillinf(self, fill=None, **fillvalues):
        newdataarrays = {datakey:_replaceinf(self.dataarrays[datakey], fillvalues.get(datakey, fill)) for datakey, dataarray in self.dataarrays.items()}
        newdataset = xr.merge([value.to_dataset(name=key) for key, value in newdataarrays.items()], join='outer') 
        newdataset.attrs = self.dataset.attrs  
        return self.__class__(newdataset, variables=self.variables.copy(), name=self.name)

    def fillneg(self, fill=None, **fillvalues):
        newdataarrays = {datakey:_replaceneg(self.dataarrays[datakey], fillvalues.get(datakey, fill)) for datakey, dataarray in self.dataarrays.items()}
        newdataset = xr.merge([value.to_dataset(name=key) for key, value in newdataarrays.items()], join='outer') 
        newdataset.attrs = self.dataset.attrs  
        return self.__class__(newdataset, variables=self.variables.copy(), name=self.name)   

    @keydispatcher
    def fillextreme(self, method, *args, threshold, fill=None, **fillvalues): raise KeyError(method)        
    @fillextreme.register('stdev', 'std')
    def __fillextreme_stdev(self, *args, threshold, fill=None, **fillvalues):
        assert isinstance(threshold, Number)
        newdataarrays = {datakey:_replacestd(dataarray, fillvalues.get(datakey, fill), threshold) for datakey, dataarray in self.dataarrays.items()}
        newdataset = xr.merge([value.to_dataset(name=key) for key, value in newdataarrays.items()], join='outer') 
        newdataset.attrs = self.dataset.attrs  
        return self.__class__(newdataset, variables=self.variables.copy(), name=self.name)       
    
    def transpose(self, *axes):
        assert all([key in self.axeskeys for key in axes])
        order = axes + tuple([key for key in self.dimkeys if key not in axes])
        newdataset = self.dataset.transpose(*order)
        newdataset.attrs = self.dataset.attrs
        return self.__class__(newdataset, variables=self.variables.copy(), name=self.name)

    def addscope(self, axis, value, variable):
        assert axis not in self.keys
        newdataset = self.dataset.assign_coords(**{axis:str(variable(value))})
        newvariables = self.variables.update({axis:variable})
        return self.__class__(newdataset, variables=newvariables, name=self.name) 

    def expand(self, axis):
        assert axis in self.scopekeys
        newdataset = self.dataset.expand_dims(axis)        
        newdataset.attrs = self.dataset.attrs
        return self.__class__(newdataset, variables=self.variables.copy(), name=self.name)
    
    def squeeze(self, *axes):
        assert all([axis in self.axeskeys for axis in axes])
        newdataset = self.dataset
        for axis in axes: 
            if axis in self.scopekeys: pass
            elif axis in self.headerkeys:
                assert len(self.dataset.coords[axis]) == 1
                newdataset = newdataset.squeeze(dim=axis, drop=False)
            else: raise ValueError(axis)
        newdataset.attrs = self.dataset.attrs
        return self.__class__(newdataset, variables=self.variables.copy(), name=self.name)

    def removescope(self, *axes):
        assert all([axis in self.scopekeys for axis in axes])
        newdataset, newvariables = self.dataset, self.variables.copy()
        for axis in axes: 
            newdataset = newdataset.drop(axis)
            newvariables.pop(axis)
        newdataset.attrs = self.dataset.attrs        
        return self.__class__(newdataset, variables=newvariables, name=self.name)

    def __mul__(self, factor): return self.multiply(factor)
    def __truediv__(self, factor): return self.divide(factor)
    
    def multiply(self, factor, *args, **kwargs):
        assert isinstance(factor, Number)
        if factor == 1: return self
        newdataset = self.dataset * factor
        newdataset.attrs = self.dataset.attrs
        newvariables = self.variables.update({datakey:self.variables[datakey].transformation(*args, method='factor', how='multiply', factor=factor, **kwargs) for datakey in _aslist(self.datakeys)})
        return self.__class__(newdataset, variables=newvariables, name=self.name)    
        
    def divide(self, factor, *args, **kwargs):
        assert isinstance(factor, Number)
        if factor == 1: return self
        newdataset = self.dataset / factor
        newdataset.attrs = self.dataset.attrs     
        newvariables = self.variables.update({datakey:self.variables[datakey].transformation(*args, method='factor', how='divide', factor=factor, **kwargs) for datakey in _aslist(self.datakeys)})
        return self.__class__(newdataset, variables=newvariables, name=self.name)    

    def flatten(self): 
        dataset = self.dataset.copy()
        dataframe = dataframe_fromxarray(dataset) 
        for datakey in self.datakeys: dataframe[datakey] = dataframe[datakey].apply(lambda x: str(self.variables[datakey](x)))
        return FlatTable(dataframe, variables=self.variables.copy(), name=self.name)     

    def tohistogram(self, *args, **kwargs): 
        assert all([self.layers == 1, self.dims == 1])
        datatype = self.variables[self.headerkeys[0]].datatype
        indexvalues, axisvalues, weightvalues = self.__tohistogram(datatype, *args, **kwargs)
        indexvalues, axisvalues, weightvalues = self.__zippedsort(indexvalues, axisvalues, weightvalues)
        histarray = HistArray(self.datakeys[0], weightvalues, self.headerkeys[0], axisvalues, indexvalues, self.scope)
        return HistTable(histarray, *args, name=self.name, variables=self.variables, **kwargs)        
        
    @staticmethod
    def __zippedsort(index, axis, weights):
        content = [(i, a, w) for i, a, w in sorted(zip(index, axis, weights), key=lambda zipped: zipped[0])]
        function = lambda x: [y[x] for y in content]
        index, axis, weights = np.array(function(0)), function(1), np.array(function(2))        
        return index, axis, weights     
        
    @keydispatcher
    def __tohistogram(self, datatype, *args, **kwargs): raise KeyError(datatype)    
    
    @__tohistogram.register('num')
    def __fromnum(self, *args, **kwargs): 
        axisvalues = self.headers[self.headerkeys[0]]
        indexvalues =  np.array([self.variables[self.headerkeys[0]].fromstr(axisvalue).value for axisvalue in axisvalues])
        weightvalues = self.arrays[self.datakeys[0]]
        return indexvalues, axisvalues, weightvalues
    
    @__tohistogram.register('range')
    def __fromrange(self, *args, how, **kwargs):
        axisvalues = self.headers[self.headerkeys[0]]
        indexvalues = np.array([self.variables[self.headerkeys[0]].fromstr(axisvalue).consolidate(*args, how=how, **kwargs).value for axisvalue in axisvalues])
        weightvalues = self.arrays[self.datakeys[0]]
        return indexvalues, axisvalues, weightvalues    
    
    @__tohistogram.register('category')
    def __fromcategory(self, *args, **kwargs):
        axisvalues = self.variables[self.headerkeys[0]].spec.categories 
        indexvalues = self.variables[self.headerkeys[0]].spec.indexes 
        b = self.arrays[self.datakeys[0]]
        a = np.zeros((len(b), len(axisvalues)))
        for i, cats in enumerate(self.headers[self.headerkeys[0]]):
            js = [indexvalues[axisvalues.index(cat)] for cat in _aslist(cats)]
            for j in js: a[i, j] = 1
        weightvalues = np.linalg.solve(a, b)
        return indexvalues, axisvalues, weightvalues
            














