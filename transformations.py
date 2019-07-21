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
__all__ = ['Scale', 'Reduction', 'Cumulate', 'Uncumulate', 'Consolidate', 'Recumulate', 'Unconsolidate', 'Interpolate', 'Inversion', 'WeightedAverage', 'Boundary']
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
        transform = self.execute(*args, **content, axis=axis, **self.hyperparms, **kwargs)
        content.update(transform)
        return TableClass(**content)
        
    @abstractmethod
    def execute(self, *args, data, variables, axis, **kwargs): pass

    @classmethod
    def register(cls, *required_hyperparms, xarray_funcs={}, varray_funcs={}, **default_hyperparms):  
        def wrapper(subclass):
            name = subclass.__name__
            bases = (subclass, cls)
            attrs = dict(default_hyperparms=default_hyperparms, required_hyperparms=[*required_hyperparms, *default_hyperparms.keys()], 
                         xarray_funcs=xarray_funcs, varray_funcs=varray_funcs)           
            return type(name, bases, attrs)
        return wrapper  


@Transformation.register('method', xarray_funcs={'normalize':xar.normalize, 'standardize':xar.standardize, 'minmax':xar.minmax})
class Scale: 
    def execute(self, *args, data, variables, datakey, axis, method, **kwargs):
        xarray = self.xarray_funcs[method](data, *args, axis=axis, **kwargs)
        variables[datakey] = variables[datakey].scale(*args, method=method, axis=variables[axis].data(), **kwargs)
        return {'data': xarray, 'variables': variables}


@Transformation.register('method', xarray_funcs={'summation':xar.summation, 'mean':xar.mean, 'stdev':xar.stdev, 'minimum':xar.minimum, 'maximum':xar.maximum, 'average':xar.average}, varray_funcs={'summation':var.summation})
class Reduction: 
    def execute(self, *args, data, variables, datakey, axis, method, **kwargs):
        varray = getheader(data, axis, variables[axis])
        xarray = self.xarray_funcs[method](data, *args, axis=axis, **kwargs)
        varray = self.varray_funcs['summation'](varray, *args, **kwargs)
        xarray.attrs.update({axis:varray})
        variables[datakey] = variables[datakey].transformation(*args, method=method, axis=axis, **kwargs)
        return {'data': xarray, 'variables': variables}
    
    
@Transformation.register(xarray_funcs={'average':xar.average}, varray_funcs={'summation':var.summation})
class WeightedAverage:
    def execute(self, *args, data, variables, datakey, axis, **kwargs):
        varray = getheader(data, axis, variables[axis])
        values = [item.value for item in varray]
        xarray = self.xarray_funcs['average'](data, *args, axis=axis, weights=values, **kwargs)
        varray = self.varray_funcs['summation'](varray, *args, **kwargs)
        xarray = setheader(xarray, axis, varray)
        variables[datakey] = variables[datakey].transformation(*args, method='average', weights=axis, **kwargs)
        return {'data': xarray, 'variables': variables}        
    
    
@Transformation.register('direction', xarray_funcs={'cumulate':xar.cumulate}, varray_funcs={'cumulate':var.cumulate})
class Cumulate: 
    def execute(self, *args, data, variables, axis, direction, **kwargs):
        varray = getheader(data, axis, variables[axis])
        xarray = self.xarray_funcs['cumulate'](data, *args, axis=axis, direction=direction, **kwargs)
        varray = self.varray_funcs['cumulate'](varray, *args, direction=direction, **kwargs)
        xarray = setheader(xarray, axis, varray)
        return {'data': xarray}
    
    
@Transformation.register('direction', xarray_funcs={'uncumulate':xar.uncumulate}, varray_funcs={'uncumulate':var.uncumulate})
class Uncumulate: 
    def execute(self, *args, data, variables, axis, direction, **kwargs):
        varray = getheader(data, axis, variables[axis])
        xarray = self.xarray_funcs['uncumulate'](data, *args, axis=axis, direction=direction, **kwargs)
        varray = self.varray_funcs['uncumulate'](varray, *args, direction=direction, **kwargs)
        xarray = setheader(xarray, axis, varray)
        return {'data': xarray}
    

@Transformation.register('method', 'period', xarray_funcs={'average':xar.movingaverage}, varray_funcs={'average':var.movingaverage, 'summation':var.movingtotal, 'range':var.movingrange})
class MovingAverage:
    def execute(self, *args, data, variables, datakey, axis, method, period, **kwargs):
        varray = getheader(data, axis, variables[axis])
        xarray = self.xarray_funcs['average'](data, *args, axis=axis, direction=direction, **kwargs)
        varray = self.varray_funcs[method](varray, *args, period=period, **kwargs)
        xarray = setheader(xarray, axis, varray)
        variables[datakey] = # (Upper|Cum_XXX)_Num_Variable or (Upper|Cum_XXX)_Num_Variable or XXX_Num_Variable or (50%wt|Avg_XXX)_Num_Variable
        variables[axis] = # (Quantiles_Households)_Num_Variable or (Quantiles_Households)_Range_Variable
        return {'data': xarray}        
        
    
@Transformation.register('method', varray_funcs={'consolidate':var.consolidate})
class Consolidate: 
    def execute(self, *args, data, variables, datakey, axis, method, **kwargs):
        varray = getheader(data, axis, variables[axis])
        varray = self.varray_funcs['consolidate'](varray, *args, method=method, **kwargs)
        xarray = setheader(data, axis, varray)
        variables[axis] = variables[axis].consolidate(*args, method=method, axis=axis, **kwargs)
        return {'data': xarray, 'variables': variables}


@Transformation.register('method', varray_funcs={'unconsolidate':var.unconsolidate})
class Unconsolidate:
    def execute(self, *args, data, variables, datakey, axis, method, **kwargs):
        varray = getheader(data, axis, variables[axis])
        varray = self.varray_funcs['unconsolidate'](varray, *args, method=method, **kwargs)
        xarray = setheader(data, axis, varray)
        variables[axis] = variables[axis].unconsolidate(*args, method=method, axis=axis, **kwargs)
        return {'data': xarray, 'variables': variables}


@Transformation.register('boundarys', varray_funcs={'boundary':var.boundary})
class Boundary:
    def execute(self, *args, data, variables, datakey, axis, boundarys, **kwargs):
        varray = getheader(data, axis, variables[axis])
        varray = self.varray_funcs['boundary'](varray, *args, boundarys=boundarys[axis], **kwargs)
        xarray = setheader(data, axis, varray)
        return {'data': xarray}

    
@Transformation.register('method', 'values', array_funcs={'interpolate':xar.interpolate}, varray_funcs={'factory':var.varray_fromvalues})
class Interpolate:
    def execute(self, *args, data, variables, datakey, axis, method, values, **kwargs):
        varray = getheader(data, axis, variables[axis])
        data.coords[axis] = [item.value for item in varray]        
        xarray = self.xarray_funcs['interpolate'](data, *args, axis=axis, values=values[axis], method=method, **kwargs)
        varray = self.varray_funcs['factory'](values, *args, variable=variables[axis], method=method, **kwargs)
        xarray = setheader(xarray, axis, varray)
        return {'data':xarray}


@Transformation.register('method', 'values', xarray_funcs={'inversion':nar.inversion, 'factory':xar.xarray_fromvalues}, varray_funcs={'factory':var.varray_fromvalues})
class Inversion:
    def execute(self, *args, data, variables, datakey, axis, method, values, **kwargs):        
        narray, axes, attrs = data.values, data.coords, data.attrs                
        varray = getheader(data, axis, variables[axis])
        header = [item.value for item in varray]        
        index = data.get_axis_num(axis)   
        narray = self.xarray_funcs['inversion'](narray, header, values[datakey], *args, index=index, axis=axis, method=method, **kwargs)
        varray = self.varray_funcs['factory'](values[datakey], *args, variable=variables[datakey], method=method, **kwargs)
        axes = ODict([(k, v) if k != axis else (datakey, [str(item) for item in varray]) for k, v in zip(axes.to_index().names, axes.to_index().levels)]) 
        axes = ODict([(k, pd.Index(v, name=k)) for k, v in axes.items()])
        xarray = self.xarray_funcs['factory'](narray, axes=axes, scope=attrs)        
        return {'data': xarray, 'datakey':axis}










