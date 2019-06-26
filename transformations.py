# -*- coding: utf-8 -*-
"""
Created on Sun Jun 2 2019
@name    Transformation Functions
@author: Jack Kriby Cook

"""

from abc import ABC, abstractmethod
import pandas as pd
from collections import OrderedDict as ODict

import utilities.xarrays as xar
import utilities.narrays as nar
import variables.varrays as var

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['Scale', 'Reduction', 'Cumulate', 'Consolidate', 'Interpolate', 'Inversion']
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
    def register(cls, xarray_funcs={}, varray_funcs={}, **hyperparms):  
        def wrapper(subclass):
            name = subclass.__name__
            bases = (subclass, cls)
            attrs = dict(default_hyperparms=hyperparms, xarray_funcs=xarray_funcs, varray_funcs=varray_funcs)           
            return type(name, bases, attrs)
        return wrapper  


@Transformation.register(xarray_funcs={'normalize':xar.normalize, 'stardardize':xar.standardize, 'minmax':xar.minmax})
class Scale: 
    def execute(self, *args, data, variables, key, axis, method, **kwargs):
        xarray = self.xarray_funcs[method](data, *args, axis=axis, **kwargs)
        variables[key] = getattr(variables[key], 'scale')(*args, method=method, **kwargs)
        return {'data': xarray, 'variables': variables}


@Transformation.register(xarray_funcs={'summation':xar.summation, 'mean':xar.mean, 'stdev':xar.stdev, 'minimum':xar.minimum, 'maximum':xar.maximum, 'average':xar.average}, varray_funcs={'summation':var.summation})
class Reduction: 
    def execute(self, *args, data, variables, key, axis, method, **kwargs):
        header = getheader(data, axis)
        varray = tovarray(header, variables[axis])
        xarray = self.xarray_funcs[method](data, *args, axis=axis, **kwargs)
        varray = self.varray_funcs['summation'](varray, *args, **kwargs)
        header = toheader(varray)
        xarray = setheader(xarray, header, axis)
        variables[key] = getattr(variables[key], 'modify')(*args, data=method, **kwargs)
        return {'data': xarray, 'variables': variables}
    
    
@Transformation.register(xarray_funcs={'cumulate':xar.cumulate}, varray_funcs={'cumulate':var.cumulate}, direction='lower')
class Cumulate: 
    def execute(self, *args, data, variables, axis, **kwargs):
        header = getheader(data, axis)
        varray = tovarray(header, variables[axis])
        xarray = self.xarray_funcs['cumulate'](data, *args, axis=axis, **kwargs)
        varray = self.varray_funcs['cumulate'](varray, *args, **kwargs)
        header = toheader(varray)
        xarray = setheader(xarray, header, axis)
        return {'data': xarray}
      
        
@Transformation.register(varray_funcs={'consolidate':var.consolidate})
class Consolidate: 
    def execute(self, *args, data, variables, key, axis, **kwargs):
        header = getheader(data, axis)
        varray = tovarray(header, variables[axis])
        varray = self.varray_funcs['consolidate'](varray, *args, **kwargs)
        header = toheader(varray)
        xarray = setheader(data, header, axis)
        variables[axis] = getattr(variables[axis], 'consolidate')(*args, **kwargs)
        return {'data': xarray, 'variables': variables}
    
       
@Transformation.register(xarray_funcs={'interpolate':xar.interpolate}, varray_funcs={'factory':var.varray_fromvalues}, method='linear', fill='extrapolate')
class Interpolate:
    def execute(self, *args, data, variables, key, axis, values, **kwargs):
        header = getheader(data, axis)
        varray = tovarray(header, variables[axis])
        data.coords[axis] = tovalues(varray)
        xarray = self.xarray_funcs['interpolate'](data, *args, axis=axis, values=values, **kwargs)
        varray = self.varray_funcs['factory'](values, *args, variable=variables[axis], **kwargs)
        header = toheader(varray)
        xarray = setheader(xarray, header, axis)
        return {'data':xarray}


@Transformation.register(xarray_funcs={'interpolate':nar.interpolate, 'factory':xar.xarray_fromvalues}, varray_funcs={'factory':var.varray_fromvalues}, method='linear', fill='extrapolate')
class Inversion:
    def execute(self, *args, data, variables, key, axis, values, **kwargs):
        narray, axes, attrs = data.values, data.coords, data.attrs                
        header = getheader(data, axis)
        varray = tovarray(header, variables[axis])
        hdrvals = tovalues(varray)
        axisnum = data.get_axis_num(axis)
        narray = self.xarray_funcs['interpolate'](narray, hdrvals, values, *args, axis=axisnum, invert=True, **kwargs)
        varray = self.varray_funcs['factory'](values, *args, variable=variables[key], **kwargs)
        header = toheader(varray)
        axes = ODict([(k, v) if k != axis else (key, header) for k, v in zip(axes.to_index().names, axes.to_index().levels)]) 
        axes = ODict([(k, pd.Index(v, name=k)) for k, v in axes.items()])
        xarray = self.xarray_funcs['factory'](narray, axes=axes, scope=attrs)
        return {'data': xarray, 'key':axis}


