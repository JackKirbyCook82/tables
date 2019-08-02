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
from utilities.dispatchers import clskey_singledispatcher as keydispatcher

import utilities.xarrays as xar
import utilities.narrays as nar
import variables.varrays as var

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
    def __init__(self, *args, **hyperparms): 
        self.hyperparms = {key:value for key, value in self.defaults.items()}
        self.hyperparms.update(hyperparms)  
        assert all([key in self.hyperparms.keys() for key in self.required])        
    def __repr__(self): return '{}(transformtype={}, extract={}, hyperparms={})'.format(self.__class__.__name__, self.transformtype, self.extract, self.hyperparms)

    def __call__(self, table, *args, **kwargs):
        TableClass = table.__class__
        dataarrays, variables, scope, name = table.dataarrays.copy(), table.variables.copy(), table.scope, table.name
        
        newdataarrays, newvariables = self.execute(dataarrays, variables, *args, **self.hyperparms, **kwargs)
        if not self.extract: dataarrays.update(newdataarrays)          
        else: dataarrays = newdataarrays
        variables.update(newvariables)  
        dataset = xr.merge([dataarray.to_dataset() for key, dataarray in dataarrays.items()])   
        dataset.attrs = scope   
        return TableClass(data=dataset, variables=variables, name=name)

    @keydispatcher('transformtype')
    def transform(self, transformtype, dataarrays, variables, *args, **kwargs): pass

    @transform.register('over_data')
    def __overdata(self, dataarrays, variables, *args, datakey, **kwargs):
        newdataarrays = {datakey:self.datatransform(*args, axis=datakey, dataarray=dataarrays[datakey], variable=variables[datakey], **kwargs)}
        newvariables = {datakey:self.datavariable(*args, variable=variables[datakey], **kwargs)}
        return newdataarrays, newvariables
            
    @transform.register('along_axis')
    def __alongaxis(self, dataarrays, variables, *args, datakey, axis, **kwargs):
        newdataarrays = {datakey:self.datatransform(*args, axis=axis, dataarray=dataarrays[datakey], variable=variables[axis], **kwargs)}
        newvariables = {datakey:self.datavariable(*args, variable=variables[datakey], **kwargs)}
        return newdataarrays, newvariables
    
    @transform.register('with_axis')
    def __withaxis(self, dataarrays, variables, *args, axis, **kwargs):
        newdataarrays = {datakey:self.datatransform(*args, axis=axis, dataarray=dataarrays[datakey], variable=variables[axis], **kwargs) for datakey, dataarray in dataarrays.items()}
        newvariables = {datakey:self.datavariable(*args, variable=variables[datakey], **kwargs) for datakey, dataarray in dataarrays.items()}
        newvariables[axis] = self.axisvariable(*args, variable=variables[axis], **kwargs) 
        return newdataarrays, newvariables
    
    @transform.register('over_axis')
    def __overaxis(self, dataarrays, variables, *args, axis, **kwargs):
        newdataarrays = {datakey:self.datatransform(*args, axis=axis, dataarray=dataarray, variable=variables[axis], **kwargs) for datakey, dataarray in dataarrays.items()}   
        newvariables = {axis:self.axisvariable(*args, variable=variables[axis], **kwargs)}
        return newdataarrays, newvariables
      
    @transform.register('against_axis')
    def __againstaxis(self, dataarrays, variables, *args, datakey, axis, **kwargs):
        newdataarrays = {datakey:self.datatransform(*args, datakey=datakey, axis=axis, dataarray=dataarrays[datakey], datavariable=variables[datakey], axisvariable=variables[axis], **kwargs)}
        newvariables = {datakey:self.datavariable(*args, variable=variables[datakey], **kwargs)}
        newvariables[axis] = self.axisvariable(*args, variable=variables[axis], **kwargs) 
        return newdataarrays, newvariables
    
    @abstractmethod
    def execute(self, dataarrays, variables, *args, **kwargs): pass
    @abstractmethod
    def datatransform(self, *args, axis, dataarray, **kwargs): pass
    def datavariable(self, *args, variable, **kwargs): return variable
    def axisvariable(self, *args, variable, **kwargs): return variable

    @classmethod
    def register(cls, transformtype, extract=False, required=(), xarray_funcs={}, varray_funcs={}, **defaults):  
        assert isinstance(required, tuple)
        assert all([isinstance(funcs, dict) for funcs in (xarray_funcs, varray_funcs)])
        
        def wrapper(subclass):                        
            def execute(self, dataarrays, variables, *args, **kwargs): 
                return self.transform(dataarrays, variables, *args, transformtype=transformtype, **kwargs)
            
            name = subclass.__name__
            bases = (subclass, cls)
            attrs = dict(transformtype=transformtype, extract=extract, execute=execute, defaults=defaults, required=required, xarray_funcs=xarray_funcs, varray_funcs=varray_funcs)           
            return type(name, bases, attrs)
        return wrapper  


@Transformation.register('over_data', required=('how',), xarray_funcs={'multiply':xar.multiply, 'divide':xar.divide})
class Factor:
    def datatransform(self, *args, axis, dataarray, variable, how, factor, **kwargs):
        return self.xarray_funcs[how](dataarray, *args, how=how, factor=factor, **kwargs)
    
    def datavariable(self, *args, variable, how, factor, **kwargs):
        return variable.factor(*args, how=how, factor=factor, **kwargs)


@Transformation.register('along_axis', required=('how',), xarray_funcs={'normalize':xar.normalize, 'standardize':xar.standardize, 'minmax':xar.minmax})
class Scale: 
    def datatransform(self, *args, axis, dataarray, variable, how, **kwargs):
        return self.xarray_funcs[how](dataarray, *args, axis=axis, **kwargs)
        
    def datavariable(self, *args, variable, how, **kwargs):
        return variable.scale(*args, how=how, **kwargs)


@Transformation.register('with_axis', required=('how',), xarray_funcs={'summation':xar.summation, 'mean':xar.mean, 'stdev':xar.stdev, 'minimum':xar.minimum, 'maximum':xar.maximum}, varray_funcs={'summation':var.summation})
class Reduction: 
    def datatransform(self, *args, axis, dataarray, variable, how, **kwargs):
        varray = getheader(dataarray, axis, variable)
        xarray = self.xarray_funcs[how](dataarray, *args, axis=axis, **kwargs)
        varray = self.varray_funcs['summation'](varray, *args, **kwargs)
        #xarray.attrs.update({axis:varray})
        return xarray
        
    def datavariable(self, *args, variable, how, **kwargs):        
        return variable.transformation(*args, method='reduction', how=how, **kwargs)
    
    
@Transformation.register('with_axis', xarray_funcs={'wtaverage':xar.weightaverage}, varray_funcs={'summation':var.summation})
class WeightedAverage:
    def datatransform(self, *args, axis, dataarray, variable, how, **kwargs):
        varray = getheader(dataarray, axis, variable)
        values = [item.value for item in varray]
        xarray = self.xarray_funcs['wtaverage'](dataarray, *args, axis=axis, weights=values, **kwargs)
        varray = self.varray_funcs['summation'](varray, *args, **kwargs)
        xarray = setheader(xarray, axis, varray)
        return xarray
        
    def datavariable(self, *args, variable, how, **kwargs):
        return variable.transformation(*args, method='reduction', how='wtaverage', **kwargs)     
    
    
@Transformation.register('with_axis', required=('direction',), xarray_funcs={'cumulate':xar.cumulate}, varray_funcs={'cumulate':var.cumulate})
class Cumulate: 
    def datatransform(self, *args, axis, dataarray, variable, direction, **kwargs):
        varray = getheader(dataarray, axis, variable)
        xarray = self.xarray_funcs['cumulate'](dataarray, *args, axis=axis, direction=direction, **kwargs)
        varray = self.varray_funcs['cumulate'](varray, *args, direction=direction, **kwargs)
        xarray = setheader(xarray, axis, varray)
        return xarray
    
    
@Transformation.register('with_axis', required=('direction',), xarray_funcs={'uncumulate':xar.uncumulate}, varray_funcs={'uncumulate':var.uncumulate})
class Uncumulate: 
    def datatransform(self, *args, axis, dataarray, variable, direction, **kwargs):
        varray = getheader(dataarray, axis, variable)
        xarray = self.xarray_funcs['uncumulate'](dataarray, *args, axis=axis, direction=direction, **kwargs)
        varray = self.varray_funcs['uncumulate'](varray, *args, direction=direction, **kwargs)
        xarray = setheader(xarray, axis, varray)
        return xarray
    

@Transformation.register('with_axis', required=('how', 'period'), xarray_funcs={'average':xar.movingaverage}, varray_funcs={'average':var.movingaverage, 'total':var.movingtotal, 'bracket':var.movingbracket, 'differential':var.movingdifferential})
class MovingAverage:
    def datatransform(self, *args, axis, dataarray, variable, how, period, **kwargs):
        varray = getheader(dataarray, axis, variable)
        xarray = self.xarray_funcs['average'](dataarray, *args, axis=axis, period=period, **kwargs)
        varray = self.varray_funcs[how](varray, *args, period=period, **kwargs)
        xarray = setheader(xarray, axis, varray)
        return xarray
        
    def datavariable(self, *args, variable, how, period, **kwargs):
        return variable.moving(*args, how='average', period=period, **kwargs)        
    def axisvariable(self, *args, variable, how, period, **kwargs):
        return variable.moving(*args, how=how, period=period, **kwargs)     
        
    
@Transformation.register('over_axis', required=('how',), varray_funcs={'consolidate':var.consolidate})
class Consolidate: 
    def datatransform(self, *args, axis, dataarray, variable, how, **kwargs):
        varray = getheader(dataarray, axis, variable)
        varray = self.varray_funcs['consolidate'](varray, *args, how=how, **kwargs)
        xarray = setheader(dataarray, axis, varray)
        return xarray
        
    def axisvariable(self, *args, variable, how, **kwargs):
        return variable.consolidate(*args, how=how, **kwargs)  


@Transformation.register('over_axis', required=('how',), varray_funcs={'unconsolidate':var.unconsolidate})
class Unconsolidate:
    def datatransform(self, *args, axis, dataarray, variable, how, **kwargs):
        varray = getheader(dataarray, axis, variable)
        varray = self.varray_funcs['unconsolidate'](varray, *args, how=how, **kwargs)
        xarray = setheader(dataarray, axis, varray)
        return xarray
        
    def axisvariable(self, *args, variable, how, **kwargs):
        return variable.unconsolidate(*args, how=how, **kwargs)  


@Transformation.register('over_axis', varray_funcs={'bound':var.bound})
class Bound:
    def datatransform(self, *args, axis, dataarray, variable, how, bounds, **kwargs):
        varray = getheader(dataarray, axis, variable)
        varray = self.varray_funcs['bound'](varray, *args, bounds=bounds, **kwargs)
        xarray = setheader(dataarray, axis, varray)
        return xarray

    
@Transformation.register('with_axis', required=('how',), xarray_funcs={'interpolate':xar.interpolate}, varray_funcs={'factory':var.varray_fromvalues})
class Interpolate:
    def datatransform(self, *args, axis, dataarray, variable, how, values, **kwargs):
        varray = getheader(dataarray, axis, variable)
        dataarray.coords[axis] = [item.value for item in varray]        
        xarray = self.xarray_funcs['interpolate'](dataarray, *args, axis=axis, values=values, how=how, **kwargs)
        varray = self.varray_funcs['factory'](values, *args, variable=variable, how=how, **kwargs)
        xarray = setheader(xarray, axis, varray)
        return xarray


@Transformation.register('against_axis', extract=True, required=('how',), xarray_funcs={'inversion':nar.inversion, 'factory':xar.xarray_fromvalues}, varray_funcs={'factory':var.varray_fromvalues})
class Inversion:
    def datatransform(self, *args, datakey, axis, dataarray, datavariable, axisvariable, how, values, **kwargs):
        narray, axes, attrs = dataarray.values, dataarray.coords, dataarray.attrs   
        varray = getheader(dataarray, axis, axisvariable)
        header = [item.value for item in varray]   
        index = dataarray.get_axis_num(axis)  
        narray = self.xarray_funcs['inversion'](narray, header, values, *args, index=index, axis=axis, how=how, **kwargs)
        varray = self.varray_funcs['factory'](values, *args, variable=datavariable, how=how, **kwargs)
        axes = ODict([(key, value) if key != axis else (datakey, [str(item) for item in varray]) for key, value in zip(axes.to_index().names, axes.to_index().levels)]) 
        axes = ODict([(key, pd.Index(value, name=key)) for key, value in axes.items()])
        xarray = self.xarray_funcs['factory']({axis:narray}, axes=axes, scope=attrs, forcedataset=False)  
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
    
    def __call__(self, table, *args, column, **kwargs):
        TableClass = table.__class__
        dataframe, variables, name = table.dataframe, table.variables.copy(), table.name
        
        newdataframe, newvariables = self.execute(dataframe, variables, *args, **self.hyperparms, **kwargs)        
        variables.update(newvariables)           
        return TableClass(data=newdataframe, variables=variables, name=name)
        
    def execute(self, dataframe, variables, *args, **kwargs): return self.transform(dataframe, variables, *args, **kwargs)    
    def transform(self, dataframe, variables, *args, column, **kwargs):
        newdataframe = self.datatransform(*args, column=column, dataframe=dataframe, variable=variables[column], **kwargs)
        newvariables = self.datavariable(*args, variable=variables[column], **kwargs)
        return newdataframe, newvariables
                
    def datatransform(self, *args, dataframe, column, variable, groups, right, **kwargs):
        dataframe[column] = dataframe[column].apply(lambda x: str(variable.fromstr(x).group(*args, groups=groups[column], right=right, **kwargs)))
        return dataframe
        
    def datavariable(self, *args, variable, **kwargs):
        return variable.unconsolidate(*args, how='group', **kwargs)























