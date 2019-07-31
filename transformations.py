# -*- coding: utf-8 -*-
"""
Created on Sun Jun 2 2019
@name    Transformation Objects
@author: Jack Kriby Cook

"""

from abc import ABC, abstractmethod
import pandas as pd
import xarray as xr
from collections import OrderedDict as ODict

import utilities.xarrays as xar
import utilities.narrays as nar
import variables.varrays as var

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['Scale', 'Reduction', 'WeightedAverage', 'Cumulate', 'Uncumulate', 'MovingAverage', 'Consolidate', 'Unconsolidate', 'Boundary', 'Interpolate', 'Inversion', 'Group']
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""


_aslist = lambda items: [items] if not isinstance(items, (list, tuple)) else list(items)

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
        dataarrays, variables, scope, name = table.dataarrays, table.variables, table.scope.copy(), table.name
        
        if axis in table.headerkeys: newdataarrays, newvariables = self.__axistransformation(dataarrays, variables, *args, axis=axis, **kwargs)
        elif axis in table.datakeys: newdataarrays, newvariables = self.__datatransformation(dataarrays, variables, *args, axis=axis, **kwargs)
        else: raise ValueError(axis)  
        
        newdataset = xr.merge([dataarray.to_dataset(name=key) for key, dataarray in newdataarrays.items()])   
        assert all([newdataarray.attrs == list(newdataarrays.values())[0].attrs for newdataarray in newdataarrays.values()])
                
        scope.update(list(newdataarrays.values())[0].attrs)
        newdataset.attrs = scope               
        return TableClass(data=newdataset, variables=newvariables, name=name)
        
    def __datatransformation(self, dataarrays, variables, *args, axis, **kwargs):
        newdataarrays = {datakey:(self.execute(*args, datakey=datakey, dataarray=dataarray, variables=variables, **self.hyperparms, **kwargs) if datakey == axis else dataarray) for datakey, dataarray in dataarrays.items()}
        newvariables = {datakey:(self.datavariable(*args, datakey=datakey, variable=variable, **self.hyperparms, **kwargs) if datakey == axis else variable) for datakey, variable in variables.items()}
        return newdataarrays, newvariables    
    
    def __axistransformation(self, dataarrays, variables, *args, axis, **kwargs):
        newdataarrays = {datakey:self.execute(*args, datakey=datakey, axis=axis, dataarray=dataarray, variables=variables, **self.hyperparms, **kwargs) for datakey, dataarray in dataarrays.items()}
        newvariables = {datakey:(self.datavariable(*args, datakey=datakey, axis=axis, variable=variable, **self.hyperparms, **kwargs) if datakey in dataarrays.keys() else variable) for datakey, variable in variables.items()}
        newvariables.update({axis:self.axisvariable(*args, axis=axis, variable=variables[axis], **self.hyperparms, **kwargs)})
        return newdataarrays, newvariables

    @abstractmethod
    def execute(self, *args, datakey, axis, dataarray, variables, **kwargs): pass
    def datavariable(self, *args, datakey, axis, variable, **kwargs): return variable
    def axisvariable(self, *args, axis, variable, **kwargs): return variable

    @classmethod
    def register(cls, *required_hyperparms, xarray_funcs={}, varray_funcs={}, **default_hyperparms):  
        def wrapper(subclass):
            name = subclass.__name__
            bases = (subclass, cls)
            attrs = dict(default_hyperparms=default_hyperparms, required_hyperparms=(*required_hyperparms, *default_hyperparms.keys()), 
                         xarray_funcs=xarray_funcs, varray_funcs=varray_funcs)           
            return type(name, bases, attrs)
        return wrapper  


@Transformation.register('how', xarray_funcs={'factor':xar.factor})
class Factor:
    def execute(self, *args, datakey, dataarray, variables, how, factors, **kwargs):
        return self.xarray_funcs['factor'](dataarray, *args, how=how, factor=factors[datakey], **kwargs)
    
    def datavariable(self, *args, datakey, variable, how, factors, **kwargs):
        return variable.factor(*args, how=how, factor=factors[datakey], **kwargs)


@Transformation.register('how', xarray_funcs={'normalize':xar.normalize, 'standardize':xar.standardize, 'minmax':xar.minmax})
class Scale: 
    def execute(self, *args, datakey, axis, dataarray, variables, how, **kwargs):
        return self.xarray_funcs[how](dataarray, *args, axis=axis, **kwargs)
        
    def datavariable(self, *args, datakey, axis, variable, how, **kwargs):
        return variable.scale(*args, how=how, **kwargs)


@Transformation.register('how', xarray_funcs={'summation':xar.summation, 'mean':xar.mean, 'stdev':xar.stdev, 'minimum':xar.minimum, 'maximum':xar.maximum}, varray_funcs={'summation':var.summation})
class Reduction: 
    def execute(self, *args, datakey, axis, dataarray, variables, how, **kwargs):
        varray = getheader(dataarray, axis, variables[axis])
        xarray = self.xarray_funcs[how](dataarray, *args, axis=axis, **kwargs)
        varray = self.varray_funcs['summation'](varray, *args, **kwargs)
        xarray.attrs.update({axis:varray})
        return xarray
        
    def datavariable(self, *args, datakey, axis, variable, how, **kwargs):        
        return variable.transformation(*args, method='reduction', how=how, **kwargs)
    
    
@Transformation.register(xarray_funcs={'wtaverage':xar.weightaverage}, varray_funcs={'summation':var.summation})
class WeightedAverage:
    def execute(self, *args, datakey, axis, dataarray, variables, how, **kwargs):
        varray = getheader(dataarray, axis, variables[axis])
        values = [item.value for item in varray]
        xarray = self.xarray_funcs['wtaverage'](dataarray, *args, axis=axis, weights=values, **kwargs)
        varray = self.varray_funcs['summation'](varray, *args, **kwargs)
        xarray = setheader(xarray, axis, varray)
        return xarray
        
    def datavariable(self, *args, datakey, axis, variable, how, **kwargs):
        return variable.transformation(*args, method='reduction', how='wtaverage', **kwargs)     
    
    
@Transformation.register('direction', xarray_funcs={'cumulate':xar.cumulate}, varray_funcs={'cumulate':var.cumulate})
class Cumulate: 
    def execute(self, *args, datakey, axis, dataarray, variables, direction, **kwargs):
        varray = getheader(dataarray, axis, variables[axis])
        xarray = self.xarray_funcs['cumulate'](dataarray, *args, axis=axis, direction=direction, **kwargs)
        varray = self.varray_funcs['cumulate'](varray, *args, direction=direction, **kwargs)
        xarray = setheader(xarray, axis, varray)
        return xarray
    
    
@Transformation.register('direction', xarray_funcs={'uncumulate':xar.uncumulate}, varray_funcs={'uncumulate':var.uncumulate})
class Uncumulate: 
    def execute(self, *args, datakey, axis, dataarray, variables, direction, **kwargs):
        varray = getheader(dataarray, axis, variables[axis])
        xarray = self.xarray_funcs['uncumulate'](dataarray, *args, axis=axis, direction=direction, **kwargs)
        varray = self.varray_funcs['uncumulate'](varray, *args, direction=direction, **kwargs)
        xarray = setheader(xarray, axis, varray)
        return xarray
    

@Transformation.register('how', 'period', xarray_funcs={'average':xar.movingaverage}, varray_funcs={'average':var.movingaverage, 'total':var.movingtotal, 'bracket':var.movingbracket, 'differential':var.movingdifferential})
class MovingAverage:
    def execute(self, *args, datakey, axis, dataarray, variables, how, period, **kwargs):
        varray = getheader(dataarray, axis, variables[axis])
        xarray = self.xarray_funcs['average'](dataarray, *args, axis=axis, period=period, **kwargs)
        varray = self.varray_funcs[how](varray, *args, period=period, **kwargs)
        xarray = setheader(xarray, axis, varray)
        return xarray
        
    def datavariable(self, *args, datakey, axis, variable, how, period, **kwargs):
        return variable.moving(*args, how='average', period=period, **kwargs)        
    def axisvariable(self, *args, axis, variable, how, period, **kwargs):
        return variable.moving(*args, how=how, period=period, **kwargs)     
        
    
@Transformation.register('how', varray_funcs={'consolidate':var.consolidate})
class Consolidate: 
    def execute(self, *args, datakey, axis, dataarray, variables, how, **kwargs):
        varray = getheader(dataarray, axis, variables[axis])
        varray = self.varray_funcs['consolidate'](varray, *args, how=how, **kwargs)
        xarray = setheader(dataarray, axis, varray)
        return xarray
        
    def axisvariable(self, *args, axis, variable, how, **kwargs):
        return variable.consolidate(*args, how=how, **kwargs)  


@Transformation.register('how', varray_funcs={'unconsolidate':var.unconsolidate})
class Unconsolidate:
    def execute(self, *args, datakey, axis, dataarray, variables, how, **kwargs):
        varray = getheader(dataarray, axis, variables[axis])
        varray = self.varray_funcs['unconsolidate'](varray, *args, how=how, **kwargs)
        xarray = setheader(dataarray, axis, varray)
        return xarray
        
    def axisvariable(self, *args, axis, variable, how, **kwargs):
        return variable.unconsolidate(*args, how=how, **kwargs)  


@Transformation.register('boundarys', varray_funcs={'boundary':var.boundary})
class Boundary:
    def execute(self, *args, datakey, axis, dataarray, variables, how, boundarys, **kwargs):
        varray = getheader(dataarray, axis, variables[axis])
        varray = self.varray_funcs['boundary'](varray, *args, boundarys=boundarys[axis], **kwargs)
        xarray = setheader(dataarray, axis, varray)
        return xarray

    
@Transformation.register('how', 'values', xarray_funcs={'interpolate':xar.interpolate}, varray_funcs={'factory':var.varray_fromvalues})
class Interpolate:
    def execute(self, *args, datakey, axis, dataarray, variables, how, values, **kwargs):
        varray = getheader(dataarray, axis, variables[axis])
        dataarray.coords[axis] = [item.value for item in varray]        
        xarray = self.xarray_funcs['interpolate'](dataarray, *args, axis=axis, values=values[axis], how=how, **kwargs)
        varray = self.varray_funcs['factory'](values, *args, variable=variables[axis], how=how, **kwargs)
        xarray = setheader(xarray, axis, varray)
        return xarray


class Inversion(object):
    required_hyperparms = ('how', 'values')
    default_hyperparms = {}
    xarray_funcs = {'inversion':nar.inversion, 'factory':xar.xarray_fromvalues}
    varray_funcs = {'factory':var.varray_fromvalues}
    
    def __init__(self, *args, **hyperparms): 
        self.hyperparms = {key:value for key, value in self.default_hyperparms.items()}
        self.hyperparms.update(hyperparms)  
        assert all([key in self.hyperparms.keys() for key in self.required_hyperparms])            
    def __repr__(self): return '{}({})'.format(self.__class__.__name__, self.hyperparms)  
    
    def __call__(self, table, *args, datakey, axis, **kwargs):
        TableClass = table.__class__
        dataarrays, variables, name = table.dataarrays, table.variables, table.name  
        newdataarray = self.execute(*args, datakey=datakey, dataarray=dataarrays[datakey], axis=axis, variables=variables, **self.hyperparms, **kwargs)
        newdataset = newdataarray.to_dataset()
        newdataset.attrs = newdataarray.attrs
        return TableClass(data=newdataset, variables=variables, name=name)

    def execute(self, *args, datakey, axis, dataarray, variables, how, values, **kwargs): 
        narray, axes, attrs = dataarray.values, dataarray.coords, dataarray.attrs   
        varray = getheader(dataarray, axis, variables[axis])
        header = [item.value for item in varray]   
        index = dataarray.get_axis_num(axis)  
        narray = self.xarray_funcs['inversion'](narray, header, values[datakey], *args, index=index, axis=axis, how=how, **kwargs)
        varray = self.varray_funcs['factory'](values[datakey], *args, variable=variables[datakey], how=how, **kwargs)
        axes = ODict([(key, value) if key != axis else (datakey, [str(item) for item in varray]) for key, value in zip(axes.to_index().names, axes.to_index().levels)]) 
        axes = ODict([(key, pd.Index(value, name=key)) for key, value in axes.items()])
        xarray = self.xarray_funcs['factory']({axis:narray}, axes=axes, scope=attrs, forcedataset=False)  
        return xarray        


class Group(object):
    required_hyperparms = ('groups',)
    default_hyperparms = {'right':True}
    
    def __init__(self, *args, **hyperparms): 
        self.hyperparms = {key:value for key, value in self.default_hyperparms.items()}
        self.hyperparms.update(hyperparms)  
        assert all([key in self.hyperparms.keys() for key in self.required_hyperparms])            
    def __repr__(self): return '{}({})'.format(self.__class__.__name__, self.hyperparms)      
    
    def __call__(self, table, *args, column, **kwargs):
        TableClass = table.__class__
        dataframe, variables, name = table.dataframe, table.variables.copy(), table.name
        newdataframe = self.execute(*args, dataframe=dataframe, column=column, variables=variables, **self.hyperparms, **kwargs)
        variables.update({column:self.variable(*args, variable=variables[column], **self.hyperparms, **kwargs)})
        return TableClass(data=newdataframe, variables=variables, name=name)
        
    def execute(self, *args, dataframe, column, variables, groups, right, **kwargs):
        dataframe[column] = dataframe[column].apply(lambda x: str(variables[column].fromstr(x).group(*args, groups=groups[column], right=right, **kwargs)))
        return dataframe
        
    def variable(self, *args, variable, **kwargs):
        return variable.unconsolidate(*args, how='group', **kwargs)























