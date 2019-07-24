# -*- coding: utf-8 -*-
"""
Created on Sun Jun 2 2019
@name    Transformation Objects
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
__all__ = []
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""


def getheader(xarray, axis, variable): return [variable.fromstr(item) for item in xarray.coords[axis].values]
def setheader(xarray, axis, header): 
    xarray.coords[axis] = pd.Index([str(item) for item in header], name=axis)
    return xarray 


class Transformation(ABC):
    def __init__(self, *args, **hyperparms): 
        self.hyperparms = {key:value for key, value in self.default_hyperparms.items()}
        self.hyperparms.update(hyperparms)   
        assert all([key in self.hyperparms.keys() for key in self.required_hyperparms])        
    def __repr__(self): return '{}({})'.format(self.__class__.__name__, self.hyperparms)
    
    def __call__(self, table, *args, axis, **kwargs):
        TableClass = table.__class__
        content = table.todict()
        content.update(self.execute(*args, **content, axis=axis, **self.hyperparms, **kwargs))
        content.update(self.variables(*args, **content, axis=axis, **self.hyperparms, **kwargs))
        return TableClass(**content)
        
    @abstractmethod
    def execute(self, *args, data, variables, datakey, axis, **kwargs): pass
    def variables(self, *args, data, variables, datakey, axis, **kwargs): return {}

    @classmethod
    def register(cls, *required_hyperparms, xarray_funcs={}, varray_funcs={}, **default_hyperparms):  
        def wrapper(subclass):
            name = subclass.__name__
            bases = (subclass, cls)
            attrs = dict(default_hyperparms=default_hyperparms, required_hyperparms=[*required_hyperparms, *default_hyperparms.keys()], 
                         xarray_funcs=xarray_funcs, varray_funcs=varray_funcs)           
            return type(name, bases, attrs)
        return wrapper  


@Transformation.register('how', xarray_funcs={'normalize':xar.normalize, 'standardize':xar.standardize, 'minmax':xar.minmax})
class Scale: 
    def execute(self, *args, data, variables, datakey, axis, how, **kwargs):
        xarray = self.xarray_funcs[how](data, *args, axis=axis, **kwargs)
        return {'data': xarray}
        
    def variables(self, *args, data, variables, datakey, axis, how, **kwargs):
        variables[datakey] = variables[datakey].scale(*args, how=how, axis=variables[axis].data(), **kwargs)
        return {'variables': variables}


@Transformation.register('how', xarray_funcs={'summation':xar.summation, 'mean':xar.mean, 'stdev':xar.stdev, 'minimum':xar.minimum, 'maximum':xar.maximum}, varray_funcs={'summation':var.summation})
class Reduction: 
    def execute(self, *args, data, variables, datakey, axis, how, **kwargs):
        varray = getheader(data, axis, variables[axis])
        xarray = self.xarray_funcs[how](data, *args, axis=axis, **kwargs)
        varray = self.varray_funcs['summation'](varray, *args, **kwargs)
        xarray.attrs.update({axis:varray})
        return {'data': xarray}
        
    def variables(self, *args, data, variables, datakey, axis, how, **kwargs):        
        variables[datakey] = variables[datakey].transformation(*args, method='reduction', how=how, axis=axis, **kwargs)
        return {'variables': variables}
    
    
@Transformation.register(xarray_funcs={'wtaverage':xar.weightaverage}, varray_funcs={'summation':var.summation})
class WeightedAverage:
    def execute(self, *args, data, variables, datakey, axis, **kwargs):
        varray = getheader(data, axis, variables[axis])
        values = [item.value for item in varray]
        xarray = self.xarray_funcs['wtaverage'](data, *args, axis=axis, weights=values, **kwargs)
        varray = self.varray_funcs['summation'](varray, *args, **kwargs)
        xarray = setheader(xarray, axis, varray)
        return {'data': xarray}
        
    def variables(self, *args, data, variables, datakey, axis, **kwargs):
        variables[datakey] = variables[datakey].transformation(*args, method='reduction', how='wtaverage', axis=axis, **kwargs)
        return {'variables': variables}        
    
    
@Transformation.register('direction', xarray_funcs={'cumulate':xar.cumulate}, varray_funcs={'cumulate':var.cumulate})
class Cumulate: 
    def execute(self, *args, data, variables, datakey, axis, direction, **kwargs):
        varray = getheader(data, axis, variables[axis])
        xarray = self.xarray_funcs['cumulate'](data, *args, axis=axis, direction=direction, **kwargs)
        varray = self.varray_funcs['cumulate'](varray, *args, direction=direction, **kwargs)
        xarray = setheader(xarray, axis, varray)
        return {'data': xarray}
    
    
@Transformation.register('direction', xarray_funcs={'uncumulate':xar.uncumulate}, varray_funcs={'uncumulate':var.uncumulate})
class Uncumulate: 
    def execute(self, *args, data, variables, datakey, axis, direction, **kwargs):
        varray = getheader(data, axis, variables[axis])
        xarray = self.xarray_funcs['uncumulate'](data, *args, axis=axis, direction=direction, **kwargs)
        varray = self.varray_funcs['uncumulate'](varray, *args, direction=direction, **kwargs)
        xarray = setheader(xarray, axis, varray)
        return {'data': xarray}
    

@Transformation.register('how', 'period', xarray_funcs={'average':xar.movingaverage}, varray_funcs={'average':var.movingaverage, 'total':var.movingtotal, 'bracket':var.movingbracket})
class MovingAverage:
    def execute(self, *args, data, variables, datakey, axis, how, period, **kwargs):
        varray = getheader(data, axis, variables[axis])
        xarray = self.xarray_funcs['average'](data, *args, axis=axis, period=period, **kwargs)
        varray = self.varray_funcs[how](varray, *args, period=period, **kwargs)
        xarray = setheader(xarray, axis, varray)
        return {'data': xarray}
        
    def variables(self, *args, data, variables, datakey, axis, how, period, **kwargs):
        variables[datakey] = variables[datakey].moving(*args, how='average', axis=axis, period=period, **kwargs)
        variables[axis] = variables[axis].moving(*args, how=how, period=period, **kwargs)
        return {'variables':variables}        
        
    
@Transformation.register('how', varray_funcs={'consolidate':var.consolidate})
class Consolidate: 
    def execute(self, *args, data, variables, datakey, axis, how, **kwargs):
        varray = getheader(data, axis, variables[axis])
        varray = self.varray_funcs['consolidate'](varray, *args, how=how, **kwargs)
        xarray = setheader(data, axis, varray)
        return {'data': xarray}
        
    def variables(self, *args, data, variables, datakey, axis, how, **kwargs):
        variables[axis] = variables[axis].consolidate(*args, how=how, **kwargs)
        return {'variables': variables}


@Transformation.register('how', varray_funcs={'unconsolidate':var.unconsolidate})
class Unconsolidate:
    def execute(self, *args, data, variables, datakey, axis, how, **kwargs):
        varray = getheader(data, axis, variables[axis])
        varray = self.varray_funcs['unconsolidate'](varray, *args, how=how, **kwargs)
        xarray = setheader(data, axis, varray)
        return {'data': xarray}
        
    def variables(self, *args, data, variables, datakey, axis, how, **kwargs):
        variables[axis] = variables[axis].unconsolidate(*args, how=how, **kwargs)
        return {'variables': variables}


@Transformation.register('boundarys', varray_funcs={'boundary':var.boundary})
class Boundary:
    def execute(self, *args, data, variables, datakey, axis, boundarys, **kwargs):
        varray = getheader(data, axis, variables[axis])
        varray = self.varray_funcs['boundary'](varray, *args, boundarys=boundarys[axis], **kwargs)
        xarray = setheader(data, axis, varray)
        return {'data': xarray}

    
@Transformation.register('how', 'values', xarray_funcs={'interpolate':xar.interpolate}, varray_funcs={'factory':var.varray_fromvalues})
class Interpolate:
    def execute(self, *args, data, variables, datakey, axis, how, values, **kwargs):
        varray = getheader(data, axis, variables[axis])
        data.coords[axis] = [item.value for item in varray]        
        xarray = self.xarray_funcs['interpolate'](data, *args, axis=axis, values=values[axis], how=how, **kwargs)
        varray = self.varray_funcs['factory'](values, *args, variable=variables[axis], how=how, **kwargs)
        xarray = setheader(xarray, axis, varray)
        return {'data':xarray}


@Transformation.register('how', 'values', xarray_funcs={'inversion':nar.inversion, 'factory':xar.xarray_fromvalues}, varray_funcs={'factory':var.varray_fromvalues})
class Inversion:
    def execute(self, *args, data, variables, datakey, axis, how, values, **kwargs):        
        narray, axes, attrs = data.values, data.coords, data.attrs                
        varray = getheader(data, axis, variables[axis])
        header = [item.value for item in varray]        
        index = data.get_axis_num(axis)   
        narray = self.xarray_funcs['inversion'](narray, header, values[datakey], *args, index=index, axis=axis, how=how, **kwargs)
        varray = self.varray_funcs['factory'](values[datakey], *args, variable=variables[datakey], how=how, **kwargs)
        axes = ODict([(k, v) if k != axis else (datakey, [str(item) for item in varray]) for k, v in zip(axes.to_index().names, axes.to_index().levels)]) 
        axes = ODict([(k, pd.Index(v, name=k)) for k, v in axes.items()])
        xarray = self.xarray_funcs['factory'](narray, axes=axes, scope=attrs)        
        return {'data': xarray, 'datakey':axis}










