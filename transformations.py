# -*- coding: utf-8 -*-
"""
Created on Sun Jun 2 2019
@name    Transformation Functions
@author: Jack Kriby Cook

"""

from abc import ABC, abstractmethod
from collections import namedtuple as ntuple
import pandas as pd
from functools import update_wrapper

import utilities.arrays as arr
import variables.arrays as var

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['Normalize', 'Standardize', 'MinMax', 'Average', 'Cumulate', 'Consolidate', 'Interpolate', 'Inversion']
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""


def with_strheader(function):
    def wrapper(varray, *args, **kwargs):
        ################################################################
        pass
    update_wrapper(wrapper, function)
    return wrapper


Axis = ntuple('Axis', 'index key')


class Transformation(ABC):
    def __init__(self, *args, **hyperparms): self.hyperparms.update(hyperparms)               
    
    def __call__(self, table, *args, axis, **kwargs):
        TableClass = table.__class__
        axis = Axis(table.axisindex(axis), table.axiskey(axis))
        xarray, specs = self.execute(table.xarray, table.specs, *args, axiskey=axis.key, **kwargs)
        return TableClass(xarray, specs=specs, name=table.name)
        
    @abstractmethod
    def execute(self, xarray, specs, *args, axiskey, **kwargs): pass

    def update_xarray(self, xarray, *args, axiskey, **kwargs):
        return arr.apply_toxarray(xarray, self.functions['xarray'], *args, axis=axiskey, **self.hyperparms, **kwargs)
    @with_strheader
    def update_varray(self, varray, *args, **kwargs):
        return var.apply_tovarray(varray, self.functions['varray'], *args, **self.hyperparms, **kwargs)

    def getheader(self, xarray, *args, axiskey, **kwargs): 
        return xarray.coords[axiskey].values
    def setheader(self, xarray, header, *args, axiskey, **kwargs): 
        xarray.coords[axiskey] = pd.Index(header, name=axiskey)
        return xarray

    @classmethod
    def register(cls, xarray=None, header=None, **hyperparms):  
        def wrapper(subclass):
            name = subclass.__name__
            bases = (subclass, cls)
            functions = {}
            if xarray: functions['xarray'] = xarray
            if header: functions['header'] = header
            attrs = dict(hyperparms=hyperparms, functions=functions)           
            return type(name, bases, attrs)
        return wrapper  


@Transformation.register(xarray=arr.normalize)
class Normalize: 
    def execute(self, xarray, specs, *args, axiskey, **kwargs):
        return self.update_xarray(xarray, *args, axiskey=axiskey, **kwargs), specs
        
@Transformation.register(xarray=arr.standardize)
class Standardize: 
    def execute(self, xarray, specs, *args, axiskey, **kwargs):
        return self.update_xarray(xarray, *args, axiskey=axiskey, **kwargs), specs

@Transformation.register(xarray=arr.minmax)
class MinMax: 
    def execute(self, xarray, specs, *args, axiskey, **kwargs):
        return self.update_xarray(xarray, *args, axiskey=axiskey, **kwargs), specs


@Transformation.register(xarray=arr.average, varray=var.summation, weights=None)
class Average: 
    def execute(self, xarray, specs, *args, axiskey, **kwargs):
        xarray = self.update_xarray(xarray, *args, axiskey=axiskey, **kwargs)
        header = self.getheader(xarray, *args, axiskey=axiskey, **kwargs)
        header = self.update_varray(header, *args, **kwargs)
        xarray = self.setheader(xarray, header, *args, axiskey=axiskey, **kwargs)
        return xarray, specs
    
    
@Transformation.register(xarray=arr.cumulate, varray=var.cumulate, direction='upper')
class Cumulate: 
    def execute(self, xarray, specs, *args, axiskey, **kwargs):
        xarray = self.update_xarray(xarray, *args, axiskey=axiskey, **kwargs)
        header = self.getheader(xarray, *args, axiskey=axiskey, **kwargs)    
        header = self.update_varray(header, *args, **kwargs)
        xarray = self.setheader(xarray, header, *args, axiskey=axiskey, **kwargs)
        return xarray, specs
    
    
@Transformation.register(varray=var.consolidate, method='average')
class Consolidate: 
    pass
    
@Transformation.register(xarray=arr.interpolate1D, kind='linear', fill='extrapolate')
class Interpolate: 
    pass
    
@Transformation.register(xarray=arr.interpolate1D, kind='linear', fill='extrapolate')
class Inversion: 
    pass








