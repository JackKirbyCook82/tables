# -*- coding: utf-8 -*-
"""
Created on Sun Jun 2 2019
@name    Transformation Functions
@author: Jack Kriby Cook

"""

from abc import ABC, abstractmethod
from collections import namedtuple as ntuple
import pandas as pd

import utilities.arrays as arr
import variables.arrays as var

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['Normalize', 'Standardize', 'MinMax', 'Average', 'Cumulate', 'Consolidate', 'Interpolate', 'Inversion']
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""


Axis = ntuple('Axis', 'index key')


def getheader(xarray, *args, axis, **kwargs): 
    return xarray.coords[axis].values

def setheader(xarray, header, *args, axis, **kwargs): 
    xarray.coords[axis] = pd.Index(header, name=axis)
    return xarray 

def tovarray(header, *args, variable, **kwargs): return [variable.fromstr(item) for item in header]
def toheader(varray, *args, **kwargs): return [str(item) for item in varray]


class Transformation(ABC):
    def __init__(self, *args, **hyperparms): self.hyperparms.update(hyperparms)               
    
    def __call__(self, table, *args, axis, **kwargs):
        TableClass = table.__class__
        axis = Axis(table.axisindex(axis), table.axiskey(axis))
        #xarray, variables = self.execute(table.xarray, table.variables, *args, axis=axis.key, **kwargs)
        #return TableClass(xarray, variables=variables, name=table.name)
        return table
        
    @abstractmethod
    def execute(self, xarray, variables, *args, axis, **kwargs): pass

    def update_xarray(self, xarray, *args, axis, **kwargs):
        return arr.apply_toxarray(xarray, self.functions['xarray'], *args, axis=axis, **self.hyperparms, **kwargs)
    def update_varray(self, varray, *args, **kwargs):
        return var.apply_tovarray(varray, self.functions['varray'], *args, **self.hyperparms, **kwargs)

    @classmethod
    def register(cls, xarray=None, header=None, **hyperparms):  
        def wrapper(subclass):
            name = subclass.__name__
            bases = (subclass, cls)
            functions = {}
            if xarray: functions['xarray'] = xarray
            if header: functions['varray'] = header
            attrs = dict(hyperparms=hyperparms, functions=functions)           
            return type(name, bases, attrs)
        return wrapper  


@Transformation.register(xarray=arr.normalize)
class Normalize: 
    def execute(self, xarray, variables, *args, axis, **kwargs):
        pass
        
@Transformation.register(xarray=arr.standardize)
class Standardize: 
    def execute(self, xarray, variables, *args, axis, **kwargs):
        pass

@Transformation.register(xarray=arr.minmax)
class MinMax: 
    def execute(self, xarray, variables, *args, axis, **kwargs):
        pass

@Transformation.register(xarray=arr.average, varray=var.summation, weights=None)
class Average: 
    def execute(self, xarray, variables, *args, axis, **kwargs):
        pass
    
@Transformation.register(xarray=arr.cumulate, varray=var.cumulate, direction='upper')
class Cumulate: 
    def execute(self, xarray, variables, *args, axis, **kwargs):
        pass
      
@Transformation.register(varray=var.consolidate, method='average')
class Consolidate: 
    pass
    
@Transformation.register(xarray=arr.interpolate1D, kind='linear', fill='extrapolate')
class Interpolate: 
    pass
    
@Transformation.register(xarray=arr.interpolate1D, kind='linear', fill='extrapolate')
class Inversion: 
    pass








