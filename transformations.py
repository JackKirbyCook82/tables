# -*- coding: utf-8 -*-
"""
Created on Sun Jun 2 2019
@name    Transformation Functions
@author: Jack Kriby Cook

"""

from abc import ABC, abstractmethod
import pandas as pd

import utilities.xarrays as xar
import variables.varrays as var

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['Scale', 'Average', 'Cumulate', 'Consolidate', 'Interpolate']
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""


def getheader(xarray, axis): 
    return xarray.coords[axis].values

def setheader(xarray, header, axis): 
    xarray.coords[axis] = pd.Index(header, name=axis)
    return xarray 

def tovarray(header, variable): return [variable.fromstr(item) for item in header]
def tovalues(varray): return [item.value for item in varray]
def toheader(varray): return [str(item) for item in varray]


class Transformation(ABC):
    def __init__(self, *args, **hyperparms): 
        self.hyperparms = {key:value for key, value in self.default_hyperparms.items()}
        self.hyperparms.update(hyperparms)           
    def __repr__(self): return '{}({})'.format(self.__class__.__name__, self.hyperparms)
    
    def __call__(self, table, *args, axis, **kwargs):
        TableClass = table.__class__
        content = table.todict()
        transform = self.execute(*args, **content, axis=axis, **self.hyperparms, **kwargs)
        content.update(transform)
        return TableClass(**content)
        
    @abstractmethod
    def execute(self, *args, data, variables, axis, **kwargs): pass

    @classmethod
    def register(cls, **hyperparms):  
        def wrapper(subclass):
            name = subclass.__name__
            bases = (subclass, cls)
            attrs = dict(default_hyperparms=hyperparms)           
            return type(name, bases, attrs)
        return wrapper  


@Transformation.register(method='normalize')
class Scale: 
    def execute(self, *args, data, variables, key, axis, **kwargs):
        xarray = xar.scale(data, *args, axis=axis, **kwargs)
        variables[key] = getattr(variables[key], 'scale')(*args, **kwargs)
        return {'data': xarray, 'variables': variables}


@Transformation.register(weights=None)
class Average: 
    def execute(self, *args, data, variables, key, axis, **kwargs):
        header = getheader(data, axis)
        varray = tovarray(header, variables[axis])
        xarray = xar.average(data, *args, axis=axis, **kwargs)
        varray = var.summation(varray, *args, **kwargs)
        header = toheader(varray)
        xarray = setheader(xarray, header, axis)
        variables[key] = getattr(variables[key], 'modify')(*args, mod='average', **kwargs)
        return {'data': xarray, 'variables': variables}
    
    
@Transformation.register(direction='lower')
class Cumulate: 
    def execute(self, *args, data, variables, axis, **kwargs):
        header = getheader(data, axis)
        varray = tovarray(header, variables[axis])
        xarray = xar.cumulate(data, *args, axis=axis, **kwargs)
        varray = var.cumulate(varray, *args, **kwargs)
        header = toheader(varray)
        xarray = setheader(xarray, header, axis)
        return {'data': xarray}
      
        
@Transformation.register()
class Consolidate: 
    def execute(self, *args, data, variables, key, axis, **kwargs):
        header = getheader(data, axis)
        varray = tovarray(header, variables[axis])
        varray = var.consolidate(varray, *args, **kwargs)
        header = toheader(varray)
        xarray = setheader(data, header, axis)
        variables[axis] = getattr(variables[axis], 'consolidate')(*args, **kwargs)
        return {'data': xarray, 'variables': variables}
    
    
@Transformation.register(method='linear', fill='extrapolate')
class Interpolate: 
    def execute(self, *args, data, variables, key, axis, values, **kwargs):
        header = getheader(data, axis)
        varray = tovarray(header, variables[axis])
        hdrvalues = tovalues(varray)
        xarray = setheader(data, hdrvalues, axis)
        xarray = xar.interpolate(data, *args, axis=axis, values=values, **kwargs)
        varray = var.varray_fromvalues(values, *args, variable=variables[axis], **kwargs)
        header = toheader(varray)
        xarray = setheader(xarray, header, axis)
        return {'data': xarray}
    
    






