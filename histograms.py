# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 2020
@name:   Histogram Objects
@author: Jack Kirby Cook

"""

import numpy as np
from scipy import stats

from utilities.strings import uppercase

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['Histogram', 'HistogramTable']
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""


_HISTFORMAT = 'HISTOGRAM: {name}\nAXIS:\n{axisname} = {axis}\nDATA:\n{dataname} = {data}\nSCOPE:\n{scope}'
_SCOPEFORMAT = '{key} = {value}'

_normalize = lambda x: x / np.sum(x)
_aslist = lambda items: [items] if not isinstance(items, (list, tuple)) else list(items)


class Histogram(object):
    def __str__(self): 
        contentstrings = dict(clsname=self.__class__.__name__, **self.todict())
        scopestrings = '\n'.join([_SCOPEFORMAT.format(key=key, value=value) for key, value in self.scope.items()])
        return _HISTFORMAT.format(**contentstrings, scope=scopestrings)
    
    def __init__(self, *args, axis, index, data, scope={}, name, axisname, dataname, **kwargs): 
        assert len(axis) == len(data) == len(index)
        self.__index, self.__axis, self.__data = np.array(index), list(axis), np.array(data)        
        self.__name, self.__axisname, self.__dataname = name, axisname, dataname
        self.__histogram = stats.rv_discrete(name=self.name, values=(self.__index, _normalize(self.__data)))
        self.__scope = scope

    @property
    def name(self): return self.__name     
    @property
    def axis(self): return self.__axis
    @property
    def index(self): return self.__index
    @property
    def data(self): return self.__data  
    @property
    def scope(self): return self.__scope
    @property
    def axisname(self): return self.__axisname
    @property
    def dataname(self): return self.__dataname

    @property
    def array(self): return np.array([np.full(data, index) for index, data in zip(np.nditer(self.index, np.nditer(self.data)))]).flatten()        
    
    def mean(self): return self.histogram.mean()
    def median(self): return self.histogram.median()
    def std(self): return self.histogram.std()
    def rstd(self): return self.std() / self.mean()
    def skew(self): return stats.skew(self.array)
    def kurtosis(self): return stats.kurtosis(self.array)
    
    def __len__(self): return len(self.data)
    def __ne__(self, other): return not self.__eq__(other)
    def __eq__(self, other):
        assert isinstance(self, type(other))
        return all([self.data == other.data, self.index == other.index, self.axis == other.axis, self.dataname == other.dataname, self.axisname == other.axisname])
       
    def todict(self): return dict(name=self.name, axisname=self.axisname, dataname=self.dataname, axis=self.axis, index=self.index, data=self.data) 
    def rename(self, name): 
        self.__name = name
        return self


class HistogramTable(dict):
    View = lambda table: None 
    @classmethod
    def factory(cls, view=None): 
        if view: cls.View = view
        return cls

    @property
    def view(self): return self.View(self)       
    def __str__(self):
        view = self.View(self)
        if view: return str(view)
        else: return '\n\n'.join([uppercase(self.name, withops=True), *[str(histogram) for histogram in self.values()]])
    
    def __init__(self, *args, name, histograms=[], **kwargs):
        assert all([isinstance(histogram, Histogram) for histogram in _aslist(histograms)])
        self.__name, histograms = name, {histogram.name:histogram for histogram in _aslist(histograms)}
        super().__init__(histograms)

























