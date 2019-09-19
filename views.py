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
__all__ = ['ArrayTableView', 'FlatTableView']
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""


_NAMEFORMAT = '{clsname}: "{name}"'
_ARRAYFORMAT = 'DATA[{index}] = {key}: {axes}\n{values}'
_HEADERFORMAT = '{key}:\n{values}'
_SCOPEFORMAT = '{key}: {values}'    
_VARIABLEFORMAT = 'VARIABLE[{index}] = {key}: {name}' 
_STRUCTUREFORMAT = 'Layers={layers}, Dims={dims}, Shape={shape}, Fields={fields}'
_DATAFRAMEFORMAT = 'DATA: \n{values}'


_flatten = lambda nesteditems: [item for items in nesteditems for item in items]
_headerkeys = lambda dataarray: tuple(dataarray.dims)
_scopekeys = lambda dataarray: tuple(set(dataarray.coords.keys()) - set(dataarray.dims))

_namestring = lambda clsname, name: _NAMEFORMAT.format(clsname=uppercase(clsname), name=name)
_structurestring = lambda structure: str(structure)
_arraystring = lambda dataindex, datakey, dataaxes, datavalues: _ARRAYFORMAT.format(index=dataindex, key=uppercase(datakey, withops=True), axes=tuple([uppercase(axis, withops=True) for axis in dataaxes]), values=datavalues)
_dataframestring = lambda dataframe: _DATAFRAMEFORMAT.format(values=dataframe)
_headerstring = lambda headerkey, headervalues: _HEADERFORMAT.format(key=uppercase(headerkey, withops=True), values=headervalues)
_scopestring = lambda scopekey, scopevalues: _SCOPEFORMAT.format(key=uppercase(scopekey, withops=True), values=scopevalues)
_variablestring = lambda variableindex, variablekey, variablevalue: _VARIABLEFORMAT.format(index=variableindex, key=uppercase(variablekey, withops=True), name=variablevalue.name())


StructureSgmts = ntuple('Structure', 'layers dims shape')
class Structure(StructureSgmts):
    @property
    def fields(self): return np.prod(self.shape)  
    def __str__(self): return _STRUCTUREFORMAT.format(**self._asdict(), fields=self.fields)


class TableViewBase(ABC):
    framechar = '='
    framelength = 100
    
    @classmethod
    def factory(cls, *args, framechar, framelength, **kwargs): 
        cls.framechar, cls.framelength = framechar, framelength
        return cls
        
    @property
    def frame(self): return self.framechar * self.framelength
    
    def __init__(self, table):
        self.__variablestrings = '\n'.join([_variablestring(variableindex, variablekey, variablevalue) for variableindex, variablekey, variablevalue in zip(range(len(table.variables)), table.variables.keys(), table.variables.values())])        
        self.__namestring = _namestring(table.__class__.__name__, table.name)
        self.__structurestring = _structurestring(Structure(table.layers, table.dims, table.shape))       
       
    @abstractmethod
    def strings(self): pass
    def __str__(self): return '\n'.join([self.frame, '\n\n'.join([self.__namestring, *self.strings, self.__variablestrings, self.__structurestring]), self.frame])
        

class ArrayTableView(TableViewBase):
    def __init__(self, arraytable):
        dataarrays = ODict([(datakey, arraytable[datakey].dropallna().sortall(ascending=True).dataarrays[datakey]) for datakey in arraytable.datakeys])   
        self.__datastrings = ODict([(datakey, _arraystring(dataindex, datakey, dataarray.dims, dataarray.values)) for dataindex, datakey, dataarray in zip(range(len(dataarrays)), dataarrays.keys(), dataarrays.values())])
        self.__headerstrings = ODict([(datakey, '\n'.join([_headerstring(dimkey, dataarray.coords[dimkey].values) for dimkey in _headerkeys(dataarray)])) for datakey, dataarray in dataarrays.items()])
        self.__scopestrings = {datakey:'\n'.join([_scopestring(nondimkey, dataarray.coords[nondimkey].values) for nondimkey in _scopekeys(dataarray)]) for datakey, dataarray in dataarrays.items()}
        assert self.__datastrings.keys() == self.__headerstrings.keys() == self.__scopestrings.keys()
        self.__datakeys = list(self.__datastrings.keys()) 
        super().__init__(arraytable)

    @property
    def strings(self): return _flatten([self.__string(datakey) for datakey in self.__datakeys])
    def __string(self, datakey): return [item for item in (self.__datastrings[datakey], self.__headerstrings[datakey], self.__scopestrings[datakey]) if item]    
        

class FlatTableView(TableViewBase):
    def __init__(self, flattable):
        dataframe = flattable.dataframe
        self.__dataframestrings = _dataframestring(dataframe)
        super().__init__(flattable)

    @property
    def strings(self): return [self.__dataframestrings]

















