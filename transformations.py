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
__all__ = ['Normalize', 'Standardize', 'MinMax', 'Average', 'Cumulate', 'Consolidate', 'Interpolate', 'Inversion']
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""


def getheader(xarray, axis): 
    return xarray.coords[axis].values

def setheader(xarray, header, axis): 
    xarray.coords[axis] = pd.Index(header, name=axis)
    return xarray 

def tovarray(header, variable): return [variable.fromstr(item) for item in header]
def toheader(varray): return [str(item) for item in varray]


class Transformation(ABC):
    def __init__(self, *args, **hyperparms): self.hyperparms.update(hyperparms)               
    
    def __call__(self, table, *args, axis, **kwargs):
        TableClass = table.__class__
        xarray = self.execute(*args, **table.todict(), axis=axis, **kwargs)
        variables = self.variables(*args, **table.todict(), axis=axis, **kwargs)
        data = table.data
        return TableClass(xarray, variables=variables, data=data, name=table.name)
        
    @abstractmethod
    def execute(self, *args, xarray, variables, axis, **kwargs): pass
    @abstractmethod
    def variables(self, *args, variables, data, axis, **kwargs): pass

    def update_xarray(self, xarray, *args, axis, **kwargs):
        return arr.apply_toxarray(xarray, self.functions['xarray'], *args, axis=axis, **self.hyperparms, **kwargs)
    def update_varray(self, varray, *args, **kwargs):
        return var.apply_tovarray(varray, self.functions['varray'], *args, **self.hyperparms, **kwargs)

    def update_datavariable(self, variables, *args, data, **kwargs):
        variables.update({data:getattr(variables[data], self.methods['data'])(*args, **kwargs)})
        return variables
    def update_axisvariable(self, variables, *args, axis, **kwargs):
        variables.update({axis:getattr(variables[axis], self.methods['axis'])(*args, **kwargs)})
        return variables

    @classmethod
    def register(cls, xarray=None, varray=None, data=None, axis=None, **hyperparms):  
        def wrapper(subclass):
            name = subclass.__name__
            bases = (subclass, cls)
            functions, methods = {}, {}
            if xarray: functions['xarray'] = xarray
            if varray: functions['varray'] = varray
            if data is not None: methods['data'] = data
            if axis is not None: methods['axis'] = axis
            attrs = dict(hyperparms=hyperparms, functions=functions, methods=methods)           
            return type(name, bases, attrs)
        return wrapper  


@Transformation.register(xarray=arr.normalize, data='normalized')
class Normalize: 
    def execute(self, *args, xarray, variables, axis, **kwargs):
        xarray = self.update_xarray(xarray, *args, axis=axis, **kwargs)
        return xarray
    
    def variables(self, *args, variables, data, axis, **kwargs):
        variables = self.update_datavariable(variables, *args, data=data, **kwargs)
        return variables
        
    
@Transformation.register(xarray=arr.standardize, data='standardized')
class Standardize: 
    pass


@Transformation.register(xarray=arr.minmax, data='minmaxed')
class MinMax: 
    pass


@Transformation.register(xarray=arr.average, varray=var.summation, data='averaged', weights=None)
class Average: 
    pass
    
    
@Transformation.register(xarray=arr.cumulate, varray=var.cumulate, direction='upper')
class Cumulate: 
    def execute(self, *args, xarray, variables, axis, **kwargs):
        header = getheader(xarray, axis)
        varray = tovarray(header, variables[axis])
        xarray = self.update_xarray(xarray, *args, axis=axis, **kwargs)
        varray = self.update_varray(varray, *args, axis=axis, **kwargs)
        header = toheader(varray)
        xarray = setheader(xarray, header, axis)
        return xarray
    
    def variables(self, *args, variables, data, axis, **kwargs): 
        return variables
      
        
@Transformation.register(varray=var.consolidate, method='average')
class Consolidate: 
    pass
    
    
@Transformation.register(xarray=arr.interpolate1D, kind='linear', fill='extrapolate')
class Interpolate: 
    pass
    
    
@Transformation.register(xarray=arr.interpolate1D, kind='linear', fill='extrapolate')
class Inversion: 
    pass








