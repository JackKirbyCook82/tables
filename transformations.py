# -*- coding: utf-8 -*-
"""
Created on Sun Jun 2 2019
@name    Transformation Objects
@author: Jack Kriby Cook

"""

from abc import ABC, abstractmethod
import pandas as pd
from collections import OrderedDict as ODict
import xarray as xr

import utilities.xarrays as xar
import utilities.narrays as nar
import variables.varrays as var

from tables.adapters import flattable_transform, arraytable_inversion, arraytable_transform

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['Scale', 'Reduction', 'WeightReduction', 'Cumulate', 'Uncumulate', 'Consolidate', 'Unconsolidate', 'Moving', 'GroupBy', 'Interpolate', 'Inversion']
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""


_aslist = lambda items: [items] if not isinstance(items, (list, tuple)) else list(items)
_union = lambda x, y: list(set(x) | set(y))
_intersection = lambda x, y: list(set(x) & set(y))

headerkeys = lambda dataarray: tuple(dataarray.dims)
scopekeys = lambda dataarray: tuple(set(dataarray.coords.keys()) - set(dataarray.dims))


def getheader(dataarray, axis, variable): return [variable.fromstr(item) for item in dataarray.coords[axis].values]
def setheader(dataarray, axis, header): 
    dataarray.coords[axis] = pd.Index([str(item) for item in header], name=axis)
    return dataarray 


class Transformation(ABC):
    def __init__(self, **hyperparms): 
        self.hyperparms = {key:value for key, value in self.defaults.items()}
        self.hyperparms.update(hyperparms)  
        assert all([key in self.hyperparms.keys() for key in self.required])      
        print('Created Transformation: {}\n'.format(str(self)))
    def __repr__(self): return '{}(hyperparms={})'.format(self.__class__.__name__, self.transformtype, self.extract, self.hyperparms)
    def __str__(self): return '{}({})'.format(self.__class__.__name__, ', '.join(['='.join([key, str(value)]) for key, value in self.hyperparms.items()]))

    @arraytable_transform
    def __call__(self, xarray, *args, axis, variables, **kwargs):
        assert isinstance(variables, dict)        
        if isinstance(xarray, xr.DataArray): 
            datakey = xarray.name
            newxarray = self.execute(xarray, *args, axis=axis, datavariable=variables[datakey], axisvariable=variables[axis], **self.hyperparms, **kwargs)
            newdatavariables = {datakey:self.datavariable(variables[xarray.name], *args, datakey=datakey, axis=axis, **self.hyperparms, **kwargs)}
        elif isinstance(xarray, xr.Dataset):
            newdataarrays = [self.execute(dataarray, *args, axis=axis, datavariable=variables[datakey], axisvariable=variables[axis], **self.hyperparms, **kwargs) for datakey, dataarray in xarray.data_vars.items()]
            newxarray = xr.merge(newdataarrays, join='outer')  
            newdatavariables = {datakey:self.datavariable(variables[datakey], *args, datakey=datakey, axis=axis, **self.hyperparms, **kwargs) for datakey, dataarray in xarray.data_vars.items()}
        else: raise TypeError(type(xarray))        
        newaxisvariables = {axis:self.axisvariable(variables[axis], *args, axis=axis, **self.hyperparms, **kwargs)}
        newvariables = {**newdatavariables, **newaxisvariables}        
        return newxarray, newvariables

    @abstractmethod
    def execute(self, dataarray, *args, datakey, axis, datavariable, axisvariable, **kwargs): pass
    def datavariable(self, variable, *args, datakey, axis, **kwargs): return variable
    def axisvariable(self, variable, *args, axis, **kwargs): return variable
    
    @classmethod
    def register(cls, required=(), defaults={}, xarray_funcs={}, varray_funcs={}, **kwargs):  
        assert isinstance(required, tuple)
        assert all([isinstance(funcs, dict) for funcs in (xarray_funcs, varray_funcs)])
        
        def wrapper(subclass):                                    
            name = subclass.__name__
            bases = (subclass, cls)
            attrs = dict(defaults=defaults, required=required, xarray_funcs=xarray_funcs, varray_funcs=varray_funcs, **kwargs)           
            return type(name, bases, attrs)
        return wrapper  
    
    
@Transformation.register(required=('how',), xarray_funcs={'normalize':xar.normalize, 'standardize':xar.standardize, 'minmax':xar.minmax})
class Scale: 
    def execute(self, dataarray, *args, axis, datavariable, axisvariable, how, **kwargs):
        xarray = self.xarray_funcs[how](dataarray, *args, axis=axis, **kwargs)
        xarray.name = dataarray.name
        return xarray
     
    def datavariable(self, variable, *args, datakey, axis, how, **kwargs): return variable.transformations(*args, method='scale', how=how, axis=axis, **kwargs)
    

@Transformation.register(required=('how', 'by'), defaults={'anchor':'summation'},
                         xarray_funcs={'summation':xar.summation, 'average':xar.average, 'stdev':xar.stdev, 'minimum':xar.minimum, 'maximum':xar.maximum}, 
                         varray_funcs={'summation':var.summation, 'average':var.average, 'minimum':var.minimum, 'maximum':var.maximum})
class Reduction: 
    def execute(self, dataarray, *args, axis, datavariable, axisvariable, how, by, **kwargs):
        varray = getheader(dataarray, axis, axisvariable)
        xarray = self.xarray_funcs[how](dataarray, *args, axis=axis, **kwargs)
        varray = self.varray_funcs[by](varray, *args, **kwargs)
        xarray = xarray.assign_coords(**{axis:str(varray)}).expand_dims(axis)   
        xarray.name = dataarray.name
        return xarray
     
    def datavariable(self, variable, *args, datakey, axis, how, **kwargs): pass #TODO
    def axisvariable(self, variable, *args, axis, by, **kwargs): pass #TODO 


@Transformation.register(required=('how', 'by'),
                         xarray_funcs={'average':xar.wtaverage, 'stdev':xar.wtstdev, 'median':xar.wtmedian}, 
                         varray_funcs={'summation':var.summation, 'average':var.average, 'minimum':var.minimum, 'maximum':var.maximum})
class WeightReduction:
    def execute(self, dataarray, *args, axis, datavariable, axisvariable, how, by, **kwargs):
        varray = getheader(dataarray, axis, axisvariable)
        values = [item.value for item in varray]
        xarray = self.xarray_funcs[how](dataarray, *args, axis=axis, weights=values, **kwargs)
        varray = self.varray_funcs[by](varray, *args,  **kwargs)
        xarray = xarray.assign_coords(**{axis:str(varray)}).expand_dims(axis) 
        xarray.name = dataarray.name
        return xarray
     
    def datavariable(self, variable, *args, datakey, axis, how, **kwargs): pass #TODO
    def axisvariable(self, variable, *args, axis, by, **kwargs): pass #TODO


@Transformation.register(required=('how', 'by'),
                         xarray_funcs={'average':xar.moving_average, 'summation':xar.moving_total},
                         varray_funcs={'average':var.moving_average, 'summation':var.moving_total, 'differential':var.moving_differential, 'minimum':var.moving_mininum, 'maximum':var.moving_maximum, 'coupling':var.moving_coupling})
class Moving:
    def execute(self, dataarray, *args, axis, datavariable, axisvariable, how, by, **kwargs):
        varray = getheader(dataarray, axis, axisvariable)
        xarray = self.xarray_funcs[how](dataarray, *args, axis=axis, **kwargs)
        varray = self.varray_funcs[by](varray, *args, **kwargs)
        xarray = setheader(xarray, axis, varray)
        xarray.name = dataarray.name
        return xarray        
    
    def datavariable(self, variable, *args, datakey, axis, how, **kwargs): pass #TODO
    def axisvariable(self, variable, *args, axis, by, **kwargs): pass #TODO

    
@Transformation.register(required=('how',), 
                         xarray_funcs={'upper':xar.upper_cumulate, 'lower':xar.lower_cumulate}, 
                         varray_funcs={'upper':var.upper_cumulate, 'lower':var.lower_cumulate})
class Cumulate: 
    def execute(self, dataarray, *args, axis, datavariable, axisvariable, how, **kwargs):
        varray = getheader(dataarray, axis, axisvariable)
        xarray = self.xarray_funcs[how](dataarray, *args, axis=axis, direction=how, **kwargs)
        varray = self.varray_funcs[how](varray, *args, direction=how, **kwargs)
        xarray = setheader(xarray, axis, varray)
        xarray.name = dataarray.name
        return xarray
    
    
@Transformation.register(required=('how',),  
                         xarray_funcs={'upper':xar.upper_uncumulate, 'lower':xar.lower_uncumulate}, 
                         varray_funcs={'upper':var.upper_uncumulate, 'lower':var.lower_uncumulate})
class Uncumulate: 
    def execute(self, dataarray, *args, axis, datavariable, axisvariable, how, **kwargs):
        varray = getheader(dataarray, axis, axisvariable)
        xarray = self.xarray_funcs[how](dataarray, *args, axis=axis, direction=how, **kwargs)
        varray = self.varray_funcs[how](varray, *args, direction=how, **kwargs)
        xarray = setheader(xarray, axis, varray)
        xarray.name = dataarray.name
        return xarray
        
    
@Transformation.register(required=('how',), varray_funcs={'consolidate':var.consolidate})
class Consolidate: 
    def execute(self, dataarray, *args, axis, datavariable, axisvariable, how, **kwargs):
        varray = getheader(dataarray, axis, axisvariable)
        varray = self.varray_funcs['consolidate'](varray, *args, how=how, **kwargs)
        xarray = setheader(dataarray, axis, varray)
        xarray.name = dataarray.name
        return xarray
     
    def axisvariable(self, variable, *args, axis, how, **kwargs): return variable.transformation(*args, method='consolidate', how=how, **kwargs)


@Transformation.register(required=('how',), varray_funcs={'unconsolidate':var.unconsolidate})
class Unconsolidate:
    def execute(self, dataarray, *args, axis, datavariable, axisvariable, how, **kwargs):
        varray = getheader(dataarray, axis, axisvariable)
        varray = self.varray_funcs['unconsolidate'](varray, *args, how=how, **kwargs)
        xarray = setheader(dataarray, axis, varray)
        xarray.name = dataarray.name
        return xarray

    def axisvariable(self, variable, *args, axis, how, **kwargs): return variable.unconsolidate(*args, method='unconsolidate', how=how, **kwargs)


@Transformation.register(required=('how', 'agg',), xarray_funcs={'groupby':xar.groupby}, 
                         varray_funcs={'bins':var.groupby_bins, 'contains':var.groupby_contains, 'overlaps':var.groupby_overlaps})
class GroupBy:
    def execute(self, dataarray, *args, axis, datavariable, axisvariable, how, agg, **kwargs):
        varray = getheader(dataarray, axis, axisvariable)
        axisgroups = self.varray_funcs[how](varray, *args, **kwargs)  
        xarray = self.xarray_funcs['groupby'](dataarray, *args, axis=axis, axisgroups=axisgroups, agg=agg, **kwargs)
        xarray.name = dataarray.name
        return xarray
    
    def datavariable(self, variable, *args, datakey, axis, agg, **kwargs): pass #TODO
    def axisvariable(self, variable, *args, axis, how, **kwargs): pass #TODO
    
    
@Transformation.register(required=('how',), defaults={'how':'linear', 'fill':{}, 'smoothing':{}},
                         xarray_funcs={'interpolate':xar.interpolate}, varray_funcs={'factory':var.varray_fromvalues})
class Interpolate:
    def execute(self, dataarray, *args, axis, datavariable, axisvariable, how, values, **kwargs):
        varray = getheader(dataarray, axis, axisvariable)
        dataarray.coords[axis] = [item.value for item in varray]        
        xarray = self.xarray_funcs['interpolate'](dataarray, *args, axis=axis, values=values, how=how, **kwargs)
        varray = self.varray_funcs['factory'](values, *args, variable=axisvariable, how=how, **kwargs)
        xarray = setheader(xarray, axis, varray)
        xarray.name = dataarray.name
        return xarray


class Inversion(object):
    required=('how',)
    defaults = {'how':'linear', 'fill':{}, 'smoothing':{}}
    xarray_funcs={'inversion':nar.inversion, 'factory':xar.xarray_fromvalues}
    varray_funcs={'factory':var.varray_fromvalues}
    
    def __init__(self, **hyperparms): 
        self.hyperparms = {key:value for key, value in self.defaults.items()}
        self.hyperparms.update(hyperparms)  
        assert all([key in self.hyperparms.keys() for key in self.required])   
        print('Created Transformation: {}\n'.format(str(self)))
    def __repr__(self): return '{}(hyperparms={})'.format(self.__class__.__name__, self.transformtype, self.extract, self.hyperparms)
    def __str__(self): return '{}({})'.format(self.__class__.__name__, ', '.join(['='.join([key, str(value)]) for key, value in self.hyperparms.items()]))


    @arraytable_inversion
    def __call__(self, dataarray, *args, axis, variables, **kwargs):
        assert isinstance(variables, dict) 
        assert isinstance(dataarray, xr.DataArray)
        datakey = dataarray.name
        newdataarray = self.execute(dataarray, *args, axis=axis, datavariable=variables[datakey], axisvariable=variables[axis], **self.hyperparms, **kwargs)
        newvariables = {datakey:variables[datakey], axis:variables[axis]}        
        return newdataarray, newvariables
    
    def execute(self, dataarray, *args, axis, datavariable, axisvariable, how, values, **kwargs):
        narray, coords, attrs = dataarray.values, dataarray.coords, dataarray.attrs   
        varray = getheader(dataarray, axis, axisvariable)
        headervalues = [item.value for item in varray]   
        index = dataarray.get_axis_num(axis)  
        
        narray = self.xarray_funcs['inversion'](narray, headervalues, values, *args, index=index, axis=axis, how=how, **kwargs)
        varray = self.varray_funcs['factory'](values, *args, variable=datavariable, how=how, **kwargs)        
        
        newheaderstrs = [str(item) for item in varray]
        assert len(newheaderstrs) == len(set(newheaderstrs))
        newheader = pd.Index(newheaderstrs, name=dataarray.name)        
        dims = ODict([(key, value) if key != axis else (newheader.name, newheader) for key, value in zip(coords.to_index().names, coords.to_index().levels)]) 
        scope = ODict([(key, coords[key]) for key in scopekeys(dataarray)])

        xarray = self.xarray_funcs['factory']({axis:narray}, dims=dims, scope=scope, attrs=attrs, forcedataset=False) 
        xarray.name = axis
        return xarray   
   
 
    













