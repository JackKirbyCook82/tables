# -*- coding: utf-8 -*-
"""
Created on Tues Apr 28 2020
@name:   Table Concept Objects
@author: Jack Kirby Cook

"""

from collections import namedtuple as ntuple

from utilities.strings import uppercase

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['concept']
__copyright__ = "Copyright 2020, Jack Kirby Cook"
__license__ = ""


_aslist = lambda items: [items] if not isinstance(items, (list, tuple)) else list(items)


def concept(name, histograms=[], curves=[]): 
    assert isinstance(histograms, list) and isinstance(curves, list)
    assert all([isinstance(field, str) for field in [*histograms, *curves]])    
    
    def __new__(cls, *args, **kwargs): 
        histograms = {histogram:kwargs.pop(histogram) for histogram in cls._histograms} 
        curves = {curve:kwargs.pop(curve) for curve in cls._curves}
        histograms = {key:value.tohistogram(*args, **kwargs) for key, value in histograms.items()}
        curves = {key:value.tocurve(*args, **kwargs) for key, value in curves.items()}
        return super().__new__(cls, **histograms, **curves)  
    
    def __getitem__(self, key): return self.__getattr__(key)
    def __hash__(self): return hash((self.__class__.__name__, *[hash(getattr(self, field)) for field in self._fields],))
    
    @property
    def fields(self): return self._fields
    @property
    def histograms(self): return self._histograms
    @property
    def curves(self): return self._curves
    def todict(self): return self._asdict()    
    
    @property
    def variables(self):
        variables = {}
        for table in self: variables.update(table.variables)
        return variables

    name = uppercase(name)
    base = ntuple(uppercase(name), ' '.join([field.lower() for field in [*histograms, *curves]]))    
    attrs = {'_histograms':histograms, '_curves':curves, 'todict':todict, 'variables':variables, 'fields':fields, 'histograms':histograms, 'curves':curves,
             '__new__':__new__, '__getitem__':__getitem__, '__hash__':__hash__}
    return type(name, (base,), attrs)



