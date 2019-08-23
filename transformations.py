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

from tables.adapters import arraytable_transform, flattable_transform

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['Scale', 'Reduction', 'WeightedAverage', 'Cumulate', 'Uncumulate', 'MovingAverage', 'Consolidate', 'Unconsolidate', 'Bound', 'Interpolate', 'Inversion', 'Group']
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""


_aslist = lambda items: [items] if not isinstance(items, (list, tuple)) else list(items)

def getheader(xarray, axis, variable): return [variable.fromstr(item) for item in xarray.coords[axis].values]
def setheader(xarray, axis, header): 
    xarray.coords[axis] = pd.Index([str(item) for item in header], name=axis)
    return xarray 


class Transformation(ABC):
    def __init__(self, **hyperparms): 
        self.hyperparms = {key:value for key, value in self.defaults.items()}
        self.hyperparms.update(hyperparms)  
        assert all([key in self.hyperparms.keys() for key in self.required])        
    def __repr__(self): return '{}(hyperparms={})'.format(self.__class__.__name__, self.transformtype, self.extract, self.hyperparms)

    @arraytable_transform
    def __call__(self, dataarray, *args, axis, variables, **kwargs):
        assert isinstance(variables, dict)
        datakey = dataarray.name
        newdataarray = self.execute(dataarray, *args, axis=axis, datavariable=variables[datakey], axisvariable=variables[axis], **self.hyperparms, **kwargs)
        newvariables = self.variables(variables, *args, datakey=datakey, axis=axis, **self.hyperparms, **kwargs)       
        return newdataarray, newvariables

    @abstractmethod
    def execute(self, dataarray, *args, datakey, axis, datavariable, axisvariable, **kwargs): pass
    def variables(self, variables, *args, datakey, axis, **kwargs): return {datakey:variables[datakey], axis:variables[axis]}    

    @classmethod
    def register(cls, required=(), xarray_funcs={}, varray_funcs={}, **defaults):  
        assert isinstance(required, tuple)
        assert all([isinstance(funcs, dict) for funcs in (xarray_funcs, varray_funcs)])
        
        def wrapper(subclass):                                    
            name = subclass.__name__
            bases = (subclass, cls)
            attrs = dict(defaults=defaults, required=required, xarray_funcs=xarray_funcs, varray_funcs=varray_funcs)           
            return type(name, bases, attrs)
        return wrapper  
    
    
@Transformation.register(required=('how',), xarray_funcs={'normalize':xar.normalize, 'standardize':xar.standardize, 'minmax':xar.minmax})
class Scale: 
    def execute(self, dataarray, *args, axis, datavariable, axisvariable, how, **kwargs):
        xarray = self.xarray_funcs[how](dataarray, *args, axis=axis, **kwargs)
        xarray.name = dataarray.name
        return xarray
        
    def variables(self, variables, *args, datakey, axis, how, **kwargs):
        return {datakey:variables[datakey].scale(*args, how=how, axis=axis, **kwargs), axis:variables[axis]} 


@Transformation.register(required=('how',), xarray_funcs={'summation':xar.summation, 'mean':xar.mean, 'stdev':xar.stdev, 'minimum':xar.minimum, 'maximum':xar.maximum}, varray_funcs={'summation':var.summation})
class Reduction: 
    def execute(self, dataarray, *args, axis, datavariable, axisvariable, how, **kwargs):
        varray = getheader(dataarray, axis, axisvariable)
        xarray = self.xarray_funcs[how](dataarray, *args, axis=axis, **kwargs)
        varray = self.varray_funcs['summation'](varray, *args, **kwargs)
        xarray = xarray.assign_coords(**{axis:str(varray)}).expand_dims(axis)   
        xarray.name = dataarray.name
        return xarray
        
    def variables(self, variables, *args, datakey, axis, how, **kwargs):       
        return {datakey:variables[datakey].transformation(*args, method='reduction', how=how, **kwargs), axis:variables[axis]}    
    
    
@Transformation.register(xarray_funcs={'wtaverage':xar.weightaverage}, varray_funcs={'summation':var.summation})
class WeightedAverage:
    def execute(self, dataarray, *args, axis, datavariable, axisvariable, **kwargs):
        varray = getheader(dataarray, axis, axisvariable)
        values = [item.value for item in varray]
        xarray = self.xarray_funcs['wtaverage'](dataarray, *args, axis=axis, weights=values, **kwargs)
        varray = self.varray_funcs['summation'](varray, *args, **kwargs)
        xarray = setheader(xarray, axis, varray)
        xarray.name = dataarray.name
        return xarray
        
    def variables(self, variables, *args, datakey, axis, how, **kwargs):
        return {datakey:variables[datakey].transformation(*args, method='reduction', how='wtaverage', **kwargs), axis:variables[axis]}       
    
    
@Transformation.register(required=('direction',), xarray_funcs={'cumulate':xar.cumulate}, varray_funcs={'cumulate':var.cumulate})
class Cumulate: 
    def execute(self, dataarray, *args, axis, datavariable, axisvariable, direction, **kwargs):
        varray = getheader(dataarray, axis, axisvariable)
        xarray = self.xarray_funcs['cumulate'](dataarray, *args, axis=axis, direction=direction, **kwargs)
        varray = self.varray_funcs['cumulate'](varray, *args, direction=direction, **kwargs)
        xarray = setheader(xarray, axis, varray)
        xarray.name = dataarray.name
        return xarray
    
    
@Transformation.register(required=('direction',), xarray_funcs={'uncumulate':xar.uncumulate}, varray_funcs={'uncumulate':var.uncumulate})
class Uncumulate: 
    def execute(self, dataarray, *args, axis, datavariable, axisvariable, direction, **kwargs):
        varray = getheader(dataarray, axis, axisvariable)
        xarray = self.xarray_funcs['uncumulate'](dataarray, *args, axis=axis, direction=direction, **kwargs)
        varray = self.varray_funcs['uncumulate'](varray, *args, direction=direction, **kwargs)
        xarray = setheader(xarray, axis, varray)
        xarray.name = dataarray.name
        return xarray
        
    
@Transformation.register(required=('how', 'period'), xarray_funcs={'average':xar.movingaverage}, varray_funcs={'average':var.movingaverage, 'total':var.movingtotal, 'bracket':var.movingbracket, 'differential':var.movingdifferential})
class MovingAverage:
    def execute(self, dataarray, *args, axis, datavariable, axisvariable, how, period, **kwargs):
        varray = getheader(dataarray, axis, axisvariable)
        xarray = self.xarray_funcs['average'](dataarray, *args, axis=axis, period=period, **kwargs)
        varray = self.varray_funcs[how](varray, *args, period=period, **kwargs)
        xarray = setheader(xarray, axis, varray)
        xarray.name = dataarray.name
        return xarray
       
    def variables(self, variables, *args, datakey, axis, how, period, **kwargs):    
        return {datakey:variables[datakey].moving(*args, how='average', period=period, **kwargs), 
                axis:variables[axis].moving(*args, how=how, period=period, **kwargs)}
       
    
@Transformation.register(required=('how',), varray_funcs={'consolidate':var.consolidate})
class Consolidate: 
    def execute(self, dataarray, *args, axis, datavariable, axisvariable, how, **kwargs):
        varray = getheader(dataarray, axis, axisvariable)
        varray = self.varray_funcs['consolidate'](varray, *args, how=how, **kwargs)
        xarray = setheader(dataarray, axis, varray)
        xarray.name = dataarray.name
        return xarray
        
    def variables(self, variables, *args, datakey, axis, how, **kwargs):
      return {datakey:variables[datakey], axis:variables[axis].consolidate(*args, how=how, **kwargs)}    


@Transformation.register(required=('how',), varray_funcs={'unconsolidate':var.unconsolidate})
class Unconsolidate:
    def execute(self, dataarray, *args, axis, datavariable, axisvariable, how, **kwargs):
        varray = getheader(dataarray, axis, axisvariable)
        varray = self.varray_funcs['unconsolidate'](varray, *args, how=how, **kwargs)
        xarray = setheader(dataarray, axis, varray)
        xarray.name = dataarray.name
        return xarray
        
    def variables(self, variables, *args, datakey, axis, how, **kwargs):
        return {datakey:variables[datakey], axis:variables[axis].unconsolidate(*args, how=how, **kwargs)}      
    

@Transformation.register(varray_funcs={'bound':var.bound})
class Bound:
    def execute(self, dataarray, *args, axis, datavariable, axisvariable, bounds, **kwargs):
        varray = getheader(dataarray, axis, axisvariable)
        varray = self.varray_funcs['bound'](varray, *args, bounds=bounds, **kwargs)
        xarray = setheader(dataarray, axis, varray)
        xarray.name = dataarray.name
        return xarray  
    
    
@Transformation.register(required=('how',), xarray_funcs={'interpolate':xar.interpolate}, varray_funcs={'factory':var.varray_fromvalues})
class Interpolate:
    def execute(self, dataarray, *args, axis, datavariable, axisvariable, how, values, **kwargs):
        varray = getheader(dataarray, axis, axisvariable)
        dataarray.coords[axis] = [item.value for item in varray]        
        xarray = self.xarray_funcs['interpolate'](dataarray, *args, axis=axis, values=values, how=how, **kwargs)
        varray = self.varray_funcs['factory'](values, *args, variable=axisvariable, how=how, **kwargs)
        xarray = setheader(xarray, axis, varray)
        xarray.name = dataarray.name
        return xarray


@Transformation.register(required=('how',), xarray_funcs={'inversion':nar.inversion, 'factory':xar.xarray_fromvalues}, varray_funcs={'factory':var.varray_fromvalues})
class Inversion:
    def execute(self, dataarray, *args, axis, datavariable, axisvariable, how, values, **kwargs):
        narray, dims, attrs = dataarray.values, dataarray.coords, dataarray.attrs   
        varray = getheader(dataarray, axis, axisvariable)
        header = [item.value for item in varray]   
        index = dataarray.get_axis_num(axis)  
        narray = self.xarray_funcs['inversion'](narray, header, values, *args, index=index, axis=axis, how=how, **kwargs)
        varray = self.varray_funcs['factory'](values, *args, variable=datavariable, how=how, **kwargs)
        dims = ODict([(key, value) if key != axis else (dataarray.name, [str(item) for item in varray]) for key, value in zip(dims.to_index().names, dims.to_index().levels)]) 
        dims = ODict([(key, pd.Index(value, name=key)) for key, value in dims.items()])
        xarray = self.xarray_funcs['factory']({axis:narray}, dims=dims, attrs=attrs, forcedataset=False)  
        xarray.name = axis
        return xarray   
   
    
class Group(object):
    transformtype = 'over_column'
    required = ('right',)
    defaults = {'right': True}
    
    def __init__(self, *args, **hyperparms): 
        self.hyperparms = {key:value for key, value in self.defaults.items()}
        self.hyperparms.update(hyperparms)  
        assert all([key in self.hyperparms.keys() for key in self.required])          
    def __repr__(self): return '{}(transformtype={}, hyperparms={})'.format(self.__class__.__name__, self.transformtype, self.hyperparms)   

    @flattable_transform
    def __call__(self, dataframe, *args, column, variables, **kwargs):
        assert isinstance(variables, dict)
        newdataframe = self.execute(dataframe, *args, column=column, variable=variables[column], **self.hyperparms, **kwargs)
        newvariables = self.variables(variables, *args, column=column, **self.hyperparms, **kwargs) 
        return newdataframe, newvariables
          
    def execute(self, dataframe, *args, column, variable, groups, right, **kwargs):
        dataframe[column] = dataframe[column].apply(lambda x: str(variable.fromstr(x).group(*args, groups=groups, right=right, **kwargs)))
        return dataframe
        
    def variables(self, variables, *args, column, **kwargs):
        return {column:variables[column].unconsolidate(*args, how='group', **kwargs)}    
    













