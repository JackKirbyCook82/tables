# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 2019
@name:   Table View Objects
@author: Jack Kirby Cook

"""

from abc import ABC, abstractmethod
import numpy as np
from collections import namedtuple as ntuple
from collections import OrderedDict as ODict

from utilities.strings import uppercase

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['ArrayTableView', 'FlatTableView', 'HistTableView', 'CurveTableView']
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""


_NAMEFORMAT = '{clsname}: "{name}"'
_ARRAYFORMAT = 'DATA[{index}] = {key}: {axes}\n{values}'
_HEADERFORMAT = '{key}:\n{values}'
_SCOPEFORMAT = '{key}: {values}'    
_VARIABLEFORMAT = 'VARIABLE[{index}] = {key}: {name}' 
_STRUCTUREFORMAT = 'Layers={layers}, Dims={dims}, Shape={shape}, Fields={fields}'
_DATAFRAMEFORMAT = 'DATA: \n{values}'
_WEIGHTSFORMAT = 'WEIGHTS = {key}: \n{values}'
_AXISFORMAT = 'AXIS = {key}: \n{values}'
_INDEXFORMAT = 'INDEX = {key}: \n{values}'
_TOTALFORMAT = 'TOTAL = {total}'
_CURVEFORMAT = '{axis} = {key}: \n{values}'


_flatten = lambda nesteditems: [item for items in nesteditems for item in items]
_filterempty = lambda items: [item for item in items if item]
_headerkeys = lambda dataarray: tuple(dataarray.dims)
_scopekeys = lambda dataarray: tuple(set(dataarray.coords.keys()) - set(dataarray.dims))

_namestring = lambda clsname, name: _NAMEFORMAT.format(clsname=uppercase(clsname), name=name)
_arraystring = lambda dataindex, datakey, dataaxes, datavalues: _ARRAYFORMAT.format(index=dataindex, key=uppercase(datakey, withops=True), axes=tuple([uppercase(axis, withops=True) for axis in dataaxes]), values=datavalues)
_headerstring = lambda headerkey, headervalues: _HEADERFORMAT.format(key=uppercase(headerkey, withops=True), values=headervalues)
_scopestring = lambda scopekey, scopevalues: _SCOPEFORMAT.format(key=uppercase(scopekey, withops=True), values=scopevalues)
_structurestring = lambda structure: str(structure)
_variablestring = lambda variableindex, variablekey, variablevalue: _VARIABLEFORMAT.format(index=variableindex, key=uppercase(variablekey, withops=True), name=variablevalue.name())
_dataframestring = lambda dataframe: _DATAFRAMEFORMAT.format(values=dataframe)
_weightstring = lambda weightskey, weights: _WEIGHTSFORMAT.format(key=uppercase(weightskey), values=weights)
_axisstring = lambda axiskey, axis: _AXISFORMAT.format(key=uppercase(axiskey), values=axis)
_indexstring = lambda axiskey, index: _INDEXFORMAT.format(key=uppercase(axiskey), values=index)
_totalstring = lambda total: _TOTALFORMAT.format(total=total)
_curvestring = lambda axis, key, values: _CURVEFORMAT.format(key=uppercase(key), values=values)

class Structure(ntuple('Structure', 'layers dims shape')):
    @property
    def fields(self): return np.prod(self.shape)  
    def __str__(self): return _STRUCTUREFORMAT.format(**self._asdict(), fields=self.fields)


class TableViewBase(ABC):
    framechar = '='
    framewidth = 100
    
    @classmethod
    def factory(cls, *args, framechar, framewidth, **kwargs): 
        cls.framechar, cls.framewidth = framechar, framewidth
        return cls
     
    def __init__(self, table): self.__table = table   
    def __str__(self): return '\n'.join([self.frame, '\n\n'.join([self.namestring, *self.strings, self.variablestrings, self.structurestring]), self.frame])    
    def __call__(self, *args, **kwargs): print(str(self))
        
    @property
    def frame(self): return self.framechar * self.framewidth 
    @property
    def namestring(self): return _namestring(self.__table.__class__.__name__, self.__table.name)
    @property
    def variablestrings(self): return '\n'.join([_variablestring(variableindex, variablekey, variablevalue) for variableindex, variablekey, variablevalue in zip(range(len(self.table.variables)), self.table.variables.keys(), self.table.variables.values())])        
    @property    
    def structurestring(self): return _structurestring(Structure(self.table.layers, self.table.dims, self.table.shape))  
    @property
    def table(self): return self.__table
    @abstractmethod
    def strings(self): pass


class CurveTableView(TableViewBase):
    @property
    def xstrings(self): return _curvestring('XAXIS', self.table.xaxis, self.table.xvalues)
    @property
    def ystrings(self): return _curvestring('YAXIS', self.table.yaxis, self.table.yvalues)
    @property
    def scopestrings(self): return '\n'.join([_scopestring(scopekey=scopekey, scopevalues=scopevalues) for scopekey, scopevalues in self.table.scope.items()])
    @property
    def strings(self): return _filterempty([self.xstrings, self.ystrings, self.scopestrings])
    

class HistTableView(TableViewBase):
    @property
    def weightstrings(self): return _weightstring(self.table.weightskey, self.table.weights)
    @property
    def axisstrings(self): return _axisstring(self.table.axiskey, self.table.axis)
    @property
    def indexstrings(self): return _indexstring(self.table.axiskey, self.table.index)    
    @property
    def scopestrings(self): return '\n'.join([_scopestring(scopekey=scopekey, scopevalues=scopevalues) for scopekey, scopevalues in self.table.scope.items()])
    @property
    def totalstrings(self): return _totalstring(self.table.total())
    @property
    def strings(self): return _filterempty([self.weightstrings, self.axisstrings, self.indexstrings, self.totalstrings, self.scopestrings])
        

class ArrayTableView(TableViewBase):
    @property
    def datakeys(self): return list(self.dataarrays.keys()) 
    @property
    def dataarrays(self): return ODict([(datakey, self.table[datakey].dropallna().sortall(ascending=True).dataarrays[datakey]) for datakey in self.table.datakeys])   
    @property
    def datastrings(self): return ODict([(datakey, _arraystring(dataindex, datakey, dataarray.dims, dataarray.values)) for dataindex, datakey, dataarray in zip(range(len(self.dataarrays)), self.dataarrays.keys(), self.dataarrays.values())])   
    
    @property
    def headerstrings(self): 
        strfunction = lambda values: np.array([str(value) for value in values])
        return ODict([(datakey, '\n'.join([_headerstring(dimkey, strfunction(dataarray.coords[dimkey].values)) for dimkey in _headerkeys(dataarray)])) for datakey, dataarray in self.dataarrays.items()])    
    @property
    def scopestrings(self): 
        strfunction = lambda value: str(value)
        return {datakey:'\n'.join([_scopestring(nondimkey, strfunction(dataarray.coords[nondimkey].values)) for nondimkey in _scopekeys(dataarray)]) for datakey, dataarray in self.dataarrays.items()}
    
    @property
    def strings(self): 
        datastrings, headerstrings, scopestrings = self.datastrings, self.headerstrings, self.scopestrings
        return _filterempty(_flatten([(datastrings[datakey], headerstrings[datakey], scopestrings[datakey]) for datakey in self.datakeys]))
        

class FlatTableView(TableViewBase):
    @property
    def strings(self): 
        dataframe = self.table.dataframe.copy()
        for column in dataframe.columns:
            dataframe.loc[:, column] = dataframe[column].apply(lambda x: str(self.table.variables[column](x)))
        return [_dataframestring(dataframe)]




        
        
        
        
        
        
        
        
        
        
        
        
