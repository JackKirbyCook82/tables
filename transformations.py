# -*- coding: utf-8 -*-
"""
Created on Sun Jun 2 2019
@name    Transformation Functions
@author: Jack Kriby Cook

"""

from abc import ABC, abstractmethod
import pandas as pd

import utilities.arrays as arr
import variables.arrays as var

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['Scale', 'Average', 'Cumulate', 'Consolidate', 'Interpolate', 'Inversion']
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

    def update_xarray(self, xarray, *args, axis, **kwargs): return arr.apply_toxarray(xarray, self.functions['xarray'], *args, axis=axis, **kwargs)
    def update_varray(self, varray, *args, **kwargs): return var.apply_tovarray(varray, self.functions['varray'], *args, **kwargs)

    @classmethod
    def register(cls, functions, **hyperparms):  
        assert isinstance(functions, dict)
        def wrapper(subclass):
            name = subclass.__name__
            bases = (subclass, cls)
            attrs = dict(default_hyperparms=hyperparms, functions=functions)           
            return type(name, bases, attrs)
        return wrapper  


@Transformation.register({'xarray':arr.scale})
class Scale: 
    def execute(self, *args, data, variables, key, axis, **kwargs):
        xarray = self.update_xarray(data, *args, axis=axis, **kwargs)
        variables[key] = getattr(variables[key], 'scale')(*args, **kwargs)
        return {'data':xarray, 'variables':variables}


@Transformation.register({'xarray':arr.average, 'varray':var.summation}, weights=None)
class Average: 
    def execute(self, *args, data, variables, key, axis, **kwargs):
        header = getheader(data, axis)
        varray = tovarray(header, variables[axis])
        xarray = self.update_xarray(data, *args, axis=axis, **kwargs)
        varray = self.update_varray(varray, *args, **kwargs)
        header = toheader(varray)
        xarray = setheader(xarray, header, axis)
        variables[key] = getattr(variables[key], 'modify')(*args, mod='average', **kwargs)
        return {'data':xarray, 'variables':variables}
    
    
@Transformation.register({'xarray':arr.cumulate, 'varray':var.cumulate}, direction='lower')
class Cumulate: 
    def execute(self, *args, data, variables, axis, **kwargs):
        header = getheader(data, axis)
        varray = tovarray(header, variables[axis])
        xarray = self.update_xarray(data, *args, axis=axis, **kwargs)
        varray = self.update_varray(varray, *args, **kwargs)
        header = toheader(varray)
        xarray = setheader(xarray, header, axis)
        return {'data':xarray}
      
        
@Transformation.register({'varray':var.consolidate})
class Consolidate: 
    def execute(self, *args, data, variables, key, axis, **kwargs):
        header = getheader(data, axis)
        varray = tovarray(header, variables[axis])
        varray = self.update_varray(varray, *args, **kwargs)
        header = toheader(varray)
        xarray = setheader(data, header, axis)
        variables[axis] = getattr(variables[axis], 'consolidate')(*args, **kwargs)
        return {'data': xarray, 'variables':variables}
    
    
@Transformation.register({'xarray':arr.interpolate1D, 'varray':var.varray_fromvalues}, kind='linear', fill='extrapolate')
class Interpolate: 
    def execute(self, *args, data, variables, key, axis, values, **kwargs):
        header = getheader(data, axis)
        varray = tovarray(header, variables[axis])
        hdrvalues = tovalues(varray)
        xarray = self.update_xarray(data, hdrvalues, values, *args, axis=axis, **kwargs)
        varray = self.update_varray(varray, *args, variables[axis], **kwargs)
        header = toheader(varray)
        xarray = setheader(xarray, header, axis)
        return {'data':xarray}
    
    
@Transformation.register({'xarray':arr.interpolate1D, 'varray':var.varray_fromvalues}, kind='linear', fill='extrapolate')
class Inversion: 
    def execute(self, *args, data, variables, key, axis, values, **kwargs):
        pass








