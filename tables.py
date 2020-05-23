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
from scipy.interpolate import interp1d
from numbers import Number
from collections import OrderedDict as ODict
from collections import namedtuple as ntuple

from utilities.dataframes import dataframe_fromxarray
from utilities.xarrays import xarray_fromdataframe, standardize, absolute
from utilities.dispatchers import key_singledispatcher as dispatcher
from utilities.dispatchers import clskey_singledispatcher as keydispatcher
from utilities.dispatchers import clstype_singledispatcher as typedispatcher
from utilities.strings import uppercase

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['ArrayTable', 'FlatTable', 'HistTable']
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""


AGGREGATIONS = {'sum':np.sum, 'avg':np.mean, 'max':np.max, 'min':np.min}

_union = lambda x, y: list(set(x) | set(y))
_intersection = lambda x, y: list(set(x) & set(y))
_aslist = lambda items: [items] if not isinstance(items, (list, tuple)) else list(items)
_filterempty = lambda items: [item for item in _aslist(items) if item]
_normalize = lambda items: np.vectorize(lambda x, t: x/t)(items, items.sum())
_removezeros = lambda items: np.array([item if item > 0 else 0 for item in items])
_curveextrapolate = lambda x, y, z: interp1d(x, y, kind='linear', bounds_error=False, fill_value=(y[np.argmin(x)], z)) 
_curvebounded = lambda x, y: interp1d(x, y, kind='linear', bounds_error=True)   

_replacenan = lambda dataarray, value: xr.where(~np.isnan(dataarray), dataarray, value)
_replaceinf = lambda dataarray, value: xr.where(~np.isinf(dataarray), dataarray, value)
_replaceneg = lambda dataarray, value: xr.where(dataarray >= 0, dataarray, value)
_replacestd = lambda dataarray, value, std: xr.where(absolute(standardize(dataarray)) < std, dataarray, value)


@dispatcher
def createcurve(method, x, y, *args, **kwargs): raise KeyError(method) 
@createcurve.register('average')
def createcurve_average(x, y, *args, weights=None, **kwargs): return _curveextrapolate(x, y, np.average(y, weights=_normalize(weights) if weights else None))
@createcurve.register('last')
def createcurve_last(x, y, *args, **kwargs): return _curveextrapolate(x, y, y[np.argmax(x)])   


class EmptyHistArrayError(Exception):
    def __init__(self, histarray): super().__init__(repr(histarray))
class InvalidCurveError(Exception):
    def __init__(self, curvearray): super().__init__(repr(curvearray))


class HistArray(ntuple('HistArray', 'weightskey weights indexkey index scope')):
    def __repr__(self): return '{}({})'.format(self.__class__.__name__, ', '.join(['='.join([field, repr(getattr(self, field))]) for field in self._fields]))   
    def __hash__(self): 
        scopevaluetuple = tuple([hash(values[()]) for values in self.scope.values()])
        return hash((self.__class__.__name__, self.weightskey, tuple(self.weights), self.indexkey, tuple(self.index), tuple(self.scope.keys()), scopevaluetuple,))    
    
    def __new__(cls, weightskey, weights, indexkey, index, scope):
        assert len(weights) == len(index)
        assert isinstance(scope, dict)        
        return super().__new__(cls, weightskey, weights, indexkey, index, scope)

    def __init__(self, *args, **kwargs):
        if self.weights.sum() == 0: raise EmptyHistArrayError(self)

    def __call__(self, size, *args, **kwargs): 
        try: weights = _normalize(self.weights)
        except ZeroDivisionError: raise ZeroDivisionError(self.weights, self.weights.sum())
        try: histogram = stats.rv_discrete(values=(self.index, weights))
        except ValueError: raise ValueError(weights, weights.sum())
        return histogram.rvs(size=(1, size))[0]   
    
    
class CurveArray(ntuple('CurveArray', 'xkey xvalues ykey yvalues scope')):
    def __repr__(self): return '{}({})'.format(self.__class__.__name__, ', '.join(['='.join([field, repr(getattr(self, field))]) for field in self._fields]))  
    def __hash__(self): 
        scopevaluetuple = tuple([hash(values[()]) for values in self.scope.values()])
        return hash((self.__class__.__name__, self.weightskey, tuple(self.weights), self.indexkey, tuple(self.index), tuple(self.scope.keys()), scopevaluetuple,))    
    
    def __new__(cls, xkey, xvalues, ykey, yvalues, scope):
        assert len(yvalues) == len(xvalues)
        assert isinstance(scope, dict)
        return super().__new__(cls, xkey, xvalues, ykey, yvalues, scope)
 
    def __init__(self, *args, **kwargs):
        if len(self.xvalues) != len(set(self.xvalues)): raise InvalidCurveError(self)
    
    def __call__(self, xvalue, *args, how=None, **kwargs):  
        if how: curve = createcurve(how, self.xvalues, self.yvalues, *args, **kwargs)
        else: curve = _curvebounded(self.xvalues, self.yvalues)
        return curve(xvalue)
    

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
 

class CurveTable(TableBase):
    def __call__(self, xvalue): return self.data(xvalue, how=self.__how)
    def __init__(self, curvearray, *args, variables, how=None, **kwargs):
        assert isinstance(curvearray, CurveArray)        
        super().__init__(curvearray, *args, variables=variables, **kwargs)   
        self.__how = how
                  
    @property
    def curvearray(self): return self.data            

    @property
    def xvalues(self): return self.data.xvalues  
    @property
    def yvalues(self): return self.data.yvalues
    @property
    def xaxis(self): return [self.variables[self.xkey].fromindex(xvalue) for xvalue in self.xvalues]        
    @property
    def yaxis(self): return [self.variables[self.ykey].fromindex(yvalue) for yvalue in self.yvalues]       
    @property
    def scope(self): return self.data.scope 
    @property
    def curve(self): return self.__curve

    @property
    def xkey(self): return self.data.xkey  
    @property
    def ykey(self): return self.data.ykey 
    @property
    def scopekeys(self): return tuple(self.data.scope.keys())
    @property
    def keys(self): return (self.data.xkey, self.data.ykey, *self.data.scope.keys())

    @property
    def layers(self): return 1
    @property
    def dims(self): return 2
    @property
    def shape(self): return (2, len(self.data.xvalues))
    
    def __hash__(self): return hash(self.data)
    def __repr__(self): return repr(self.data)
    def __len__(self): return len(self.data.xvalues)       
    def __iter__(self): 
        for xvalue, xaxis, yvalue, yaxis in zip(self.xvalues, self.xaxis, self.yvalues, self.yaxis):
            yield (xvalue, xaxis), (yvalue, yaxis)   
    
    def retag(self, **tags): 
        variables, scope = self.variables.copy(), self.scope.copy()
        for oldkey, newkey in tags.items(): variables[newkey], scope[newkey] = variables[oldkey], scope[oldkey]
        data = CurveArray(tags.get(self.data.xkey, self.data.xkey), self.data.xvalues, tags.get(self.data.ykey, self.data.ykey), self.data.yvalues, scope)
        return self.__class__(data, variables=variables, name=self.name)    
    
    
class HistTable(TableBase):
    def __call__(self, size): return self.data(size)
    def __init__(self, histarray, *args, variables, **kwargs):
        assert isinstance(histarray, HistArray)        
        super().__init__(histarray, *args, variables=variables, **kwargs)    
         
    @property
    def histarray(self): return self.data
    
    @property
    def weights(self): return self.data.weights
    @property
    def index(self): return self.data.index           
    @property
    def axis(self): return [self.variables[self.indexkey].fromindex(index) for index in self.index]   
    @property
    def scope(self): return self.data.scope
    @property
    def histogram(self): return self.__histogram
    
    @property
    def weightskey(self): return self.data.weightskey
    @property
    def indexkey(self): return self.data.indexkey
    @property
    def axiskey(self): return self.data.indexkey
    @property
    def scopekeys(self): return tuple(self.data.scope.keys())
    @property
    def keys(self): return (self.data.weightskey, self.data.indexkey, *self.data.scope.keys())

    @property
    def layers(self): return 1
    @property
    def dims(self): return 2
    @property
    def shape(self): return (2, len(self.data.weights))
    
    def __getitem__(self, key): 
        if key in self.axis: return int(self.weights[list(self.axis).index(key)])
        elif key in self.index: return int(self.weights[list(self.index).index(key)])
        else: raise ValueError(key)

    def __hash__(self): return hash(self.data)
    def __repr__(self): return repr(self.data)     
    def __len__(self): return len(self.data.weights)       
    def __iter__(self): 
        for axis, index, weight in zip(self.axis, self.index, self.weights):
            yield axis, index, weight
            
    def retag(self, **tags): 
        variables, scope = self.variables.copy(), self.scope.copy()
        for oldkey, newkey in tags.items(): variables[newkey], scope[newkey] = variables[oldkey], scope[oldkey]
        data = HistArray(tags.get(self.data.weightskey, self.data.weightskey), self.data.weights, tags.get(self.data.axiskey, self.data.axiskey), self.data.axis, scope)
        return self.__class__(data, variables=variables, name=self.name)

    @property
    def array(self): return np.array([np.full(weight, index) for index, weight in zip(self.index, self.weights)]).flatten()   
    def total(self): return np.round(np.sum(self.weights), decimals=2)
    def mean(self): return self.histogram.mean()
    def median(self): return self.histogram.median()
    def std(self): return self.histogram.std()
    def rstd(self): return self.std() / self.mean()
    def skew(self): return stats.skew(self.array)
    def kurtosis(self): return stats.kurtosis(self.array)    

    def xmin(self): return np.minimum(self.index)
    def xmax(self): return np.maximum(self.index)
    def xdev(self, x): 
        if isinstance(x, Number): pass
        elif isinstance(x, str):
            if x in self.index: x = self.index.index(x) 
            else: raise ValueError(x)
        else: raise TypeError(type(x))        
        indexfunction = lambda i: pow(x - i, 2) / pow(self.xmax() - self.xmin(), 2)
        weightfunction = lambda weight: weight / self.total()
        return np.sum(np.array([indexfunction(i) * weightfunction(w) for i, w in zip(self.index, self.weights)]))
    

class FlatTable(TableBase):
    def __init__(self, data, *args, variables, **kwargs):
        try: dataframe = data.to_frame()
        except: dataframe = data    
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
        wrapper = lambda item: function(variables[axis](item))
        dataframe[key] = dataframe[axis].apply(wrapper)          
        variables[key] = type(dataframe[key].loc[0])
        return dict(data=dataframe, variables=variables, name=self.name)
    @createdata.register('multiple')
    def __createdata_fromcolumns(self, key, *args, axes, function, **kwargs):
        dataframe, variables = self.dataframe.copy(), self.variables.copy()
        wrapper = lambda items: function(*[variables[axis](items[index]) for axis, index in zip(axes, range(len(axes)))])
        dataframe[key] = dataframe[axes].apply(wrapper, axis=1)    
        variables[key] = type(dataframe[key].loc[0])
        return dict(data=dataframe, variables=variables, name=self.name)    

    def toexcel(self, file):
        if not os.path.isfile(file): openpyxl.Workbook().save(file)            
        writer = pd.ExcelWriter(file, engine='openpyxl') 
        writer.book = openpyxl.load_workbook(file)
        self.dataframe.to_excel(writer, sheet_name=self.name, index=False, header=True, engine='openpyxl') 
        writer.save()
        writer.close()

    def unflatten(self, *datakeys, **kwargs):
        dataframe = self.dataframe.copy(deep=True) 
        for column in set(dataframe.columns) - set(datakeys): dataframe[column] = dataframe[column].apply(lambda x: self.variables[column](x))            
        try: xarray = xarray_fromdataframe(dataframe, datakeys=datakeys, forcedataset=True, **kwargs)
        except: xarray = self.__unflatten(dataframe, *datakeys, **kwargs)
        arraytable = ArrayTable(xarray, variables=self.variables.copy(), name=self.name)   
        return arraytable

    def __unflatten(self, dataframe, *datakeys, **kwargs):
        for column in set(dataframe.columns) - set(datakeys): dataframe[column] = dataframe[column].apply(lambda x: x.value)  
        xarray = xarray_fromdataframe(dataframe, datakeys=datakeys, forcedataset=True, **kwargs)
        for column in set(dataframe.columns) - set(datakeys):
            varray = [self.variables[column](value) for value in xarray.coords[column].values]                               
            xarray.coords[column] = pd.Index(varray, name=column)  
        return xarray


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

    def reaxis(self, fromaxis, toaxis, values, variables, *args, **kwargs):
        try: values = {self.variables[fromaxis].fromindex(x):variables[toaxis].fromindex(y) for x, y in values.items()} 
        except: 
            try: values = {self.variables[fromaxis].fromstr(x):variables[toaxis].fromstr(y) for x, y in values.items()} 
            except: values = {self.variables[fromaxis](x):variables[toaxis](y) for x, y in values.items()}        
        newdataset = self.dataset.sel({fromaxis:list(values.keys())}).rename(name_dict={fromaxis:toaxis})
        newvariables = self.variables.copy()    
        newvariables[toaxis] = variables[toaxis]
        newdataset.coords[toaxis] = pd.Index(list(values.values()), name=toaxis)  
        return self.__class__(newdataset, variables=newvariables, name=kwargs.get('name', self.name))        
        
    def __getitem__(self, items): return self.__getitem(items)
    @typedispatcher
    def __getitem(self, items): raise TypeError(type(items))    

    @__getitem.register(int)
    def __getitemInt(self, items): return self[self.datakeys[items]] 
    @__getitem.register(str)
    def __getitemStr(self, items): return self[[items]]    
    @__getitem.register(list)
    def __getitemList(self, items): 
        newdataset = self.dataset[[item for item in items if item in self.datakeys]]
        newdataset.attrs = self.dataset.attrs
        return self.__class__(newdataset, variables=self.variables.copy(), name=self.name)
        
    def isel(self, **axes):
        axes = {key:axes.get(key, slice(None)) for key in self.headerkeys}
        newdataset = self.dataset.isel(**axes)
        for axis in axes:
            try: newdataset = newdataset.expand_dims(axis) 
            except: pass
        newdataset.attrs = self.dataset.attrs
        return self.__class__(newdataset, variables=self.variables.copy(), name=self.name) 
    
    def sel(self, **axes):
        axes = {key:list(self.headers[key]).index(value) for key, value in axes.items()}
        return self.isel(**axes)
 
    def vsel(self, **axes):
        axes = {key:axes.get(key, [header.value for header in self.headers[key]]) for key in self.headerkeys}
        axes = {key:[self.variables[key](value) for value in values] for key, values in axes.items()}
        newdataset = self.dataset.sel(**axes)
        for axis in axes:
            try: newdataset = newdataset.expand_dims(axis) 
            except: pass
        newdataset.attrs = self.dataset.attrs
        return self.__class__(newdataset, variables=self.variables.copy(), name=self.name) 
    
    def xsel(self, **axes):
        axes = {key:axes.get(key, [header.index for header in self.headers[key]]) for key in self.headerkeys}
        axes = {key:[self.variables[key].fromindex(value) for value in values] for key, values in axes.items()}
        newdataset = self.dataset.sel(**axes)
        for axis in axes:
            try: newdataset = newdataset.expand_dims(axis) 
            except: pass
        newdataset.attrs = self.dataset.attrs
        return self.__class__(newdataset, variables=self.variables.copy(), name=self.name) 
    
    def sort(self, axis, ascending=True):
        assert axis in self.dimkeys
        newdataset = self.dataset.copy()
        newdataset = newdataset.sortby(axis, ascending=ascending)
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
        newdataset = self.dataset.assign_coords(**{axis:value})
        newvariables = self.variables.update({axis:variable})
        return self.__class__(newdataset, variables=newvariables, name=self.name) 

    def expand(self, axis):
        assert axis in self.scopekeys
        newdataset = self.dataset.expand_dims(axis)        
        newdataset.attrs = self.dataset.attrs
        return self.__class__(newdataset, variables=self.variables.copy(), name=self.name)
    
    def squeeze(self, axis):
        assert axis in self.axeskeys 
        newdataset = self.dataset
        if axis in self.scopekeys: pass
        elif axis in self.headerkeys:
            if len(self.dataset.coords[axis]) != 1: raise ValueError(axis)
            newdataset = newdataset.squeeze(dim=axis, drop=False)
        else: raise ValueError(axis)
        newdataset.attrs = self.dataset.attrs
        return self.__class__(newdataset, variables=self.variables.copy(), name=self.name)

    def removescope(self, axis):
        assert axis in self.scopekeys 
        newdataset, newvariables = self.dataset, self.variables.copy()
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
        for key in self.headerkeys: dataset.coords[key] = pd.Index([item.value for item in self.dataset.coords[key].values], name=key)
        for key in self.scopekeys: dataset.coords[key] = self.dataset.coords[key]
        dataframe = dataframe_fromxarray(dataset) 
        return FlatTable(dataframe, variables=self.variables.copy(), name=self.name)        

    def tohistogram(self, *args, **kwargs): 
        assert all([self.layers == 1, self.dims == 1])
        datatype = self.variables[self.headerkeys[0]].datatype
        indexvalues, weightvalues = self.__tohistogram(datatype, *args, **kwargs)
        weightvalues = _removezeros(weightvalues)
        indexvalues, weightvalues = self.__zippedsort(indexvalues, weightvalues)
        histarray = HistArray(self.datakeys[0], weightvalues, self.headerkeys[0], indexvalues, self.scope)
        return HistTable(histarray, *args, name=self.name, variables=self.variables, **kwargs)        
        
    @staticmethod
    def __zippedsort(index, weights):
        content = [(i, w) for i, w in sorted(zip(index, weights), key=lambda zipped: zipped[0])]
        function = lambda x: [y[x] for y in content]
        index, weights = np.array(function(0)), np.array(function(1))        
        return index, weights     
        
    @keydispatcher
    def __tohistogram(self, datatype, *args, **kwargs): 
        indexvalues = np.array([header.value for header in self.headers[self.headerkeys[0]]])        
        weightvalues = self.arrays[self.datakeys[0]]
        return indexvalues, weightvalues        
    
    @__tohistogram.register('range')
    def __histogramFromRange(self, *args, how, **kwargs):
        indexvalues = np.array([header.consolidate(*args, how=how, **kwargs).value for header in self.headers[self.headerkeys[0]]])
        weightvalues = self.arrays[self.datakeys[0]]
        return indexvalues, weightvalues     
    
    @__tohistogram.register('category')
    def __histogramFromCategory(self, *args, **kwargs):
        axisvalues = self.variables[self.headerkeys[0]].spec.categories 
        indexvalues = self.variables[self.headerkeys[0]].spec.indexes 
        b = self.arrays[self.datakeys[0]]
        a = np.zeros((len(b), len(axisvalues)))
        for i, cats in enumerate(self.headers[self.headerkeys[0]]):
            js = [indexvalues[axisvalues.index(str(cat))] for cat in _aslist(cats)]
            for j in js: a[i, j] = 1
        weightvalues = np.linalg.solve(a, b)
        return indexvalues, weightvalues
           
    def tocurve(self, *args, **kwargs):
        assert all([self.layers == 1, self.dims == 1])
        datatype = self.variables[self.headerkeys[0]].datatype
        xvalues, yvalues = self.__tocurve(datatype, *args, **kwargs)        
        curvearray = CurveArray(self.headerkeys[0], xvalues, self.datakeys[0], yvalues, self.scope)
        return CurveTable(curvearray, *args, name=self.name, variables=self.variables, **kwargs)  
    
    @keydispatcher
    def __tocurve(self, datatype, *args, **kwargs): raise KeyError(datatype)
    
    @__tocurve.register('num')
    def __curveFromNum(self, *args, **kwargs):
        xvalues = np.array([header.value for header in self.headers[self.headerkeys[0]]])
        yvalues = self.arrays[self.datakeys[0]]
        return xvalues, yvalues
    
    @__tocurve.register('range')
    def __curveFromRange(self, *args, how, **kwargs):
        xvalues = np.array([header.consolidate(*args, how=how, **kwargs).value for header in self.headers[self.headerkeys[0]]])             
        yvalues = self.arrays[self.datakeys[0]]
        return xvalues, yvalues
    
    @__tocurve.register('date', 'datetime')
    def __curveFromDate(self, *args, **kwargs):
        xvalues = np.array([header.index for header in self.headers[self.headerkeys[0]]])             
        yvalues = self.arrays[self.datakeys[0]]
        return xvalues, yvalues 










