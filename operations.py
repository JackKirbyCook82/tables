# -*- coding: utf-8 -*-
"""
Created on Sun Jun 2 2019
@name    Operation Functions
@author: Jack Kriby Cook

"""

from functools import update_wrapper
import pandas as pd
import xarray as xr

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['multiply', 'divide', 'combine', 'merge', 'append']
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""


OPERATIONS = {}

_aslist = lambda items: [items] if not isinstance(items, (list, tuple)) else list(items)


def operation(*args, **kwargs):
    def decorator(function):
        def wrapper(table, other, *wargs, datakey, otherdatakey, **wkwargs):
            TableClass = table.__class__       
            dataset, dataarray, variables, name = table.dataset, table.dataarrays[datakey], table.variables.copy(), table.name
            otherdataarray, othervariables, othername = other.dataarrays[otherdatakey], other.variables, other.name
            newdataarray = function(dataarray, otherdataarray, *wargs, **wkwargs)
            try: newdatakey = kwargs['newdatakey'](datakey, otherdatakey)
            except KeyError: newdatakey = datakey            
            try: newvariables = {newdatakey:kwargs['newvariable'](variables[datakey], othervariables[otherdatakey], wargs, wkwargs)}
            except KeyError: newvariables = {}  
            try: newname = wkwargs.get('name', kwargs['newname'](name, othername))
            except KeyError: newname = name
            variables.update(newvariables)
            newdataset = newdataarray.to_dataset(name=newdatakey)  
            newdataset.attrs = dataset.attrs            
            return TableClass(data=newdataset, variables=variables, name=newname)
      
        update_wrapper(wrapper, function)
        OPERATIONS[function.__name__] = wrapper
        return wrapper
    return decorator


@operation(newdatakey = lambda tablekey, otherkey: '*'.join([tablekey, otherkey]),
           newname = lambda tablename, othername: '*'.join([tablename, othername]),
           newvariable = lambda tablevar, othervar, args, kwargs: tablevar.operation(othervar, *args, method='multiply', **kwargs))
def multiply(dataarray, otherdataarray, *args, **kwargs): 
    assert all([dataarray.attrs[key] == otherdataarray.attrs[key] for key in dataarray.attrs.keys() if key in otherdataarray.attrs.keys()])       
    newdataarray = dataarray * otherdataarray
    scope = dataarray.attrs
    scope.update(otherdataarray.attrs)
    newdataarray.attrs = scope
    newdataarray.name = dataarray.name
    return newdataarray


@operation(newdatakey = lambda tablekey, otherkey: '/'.join([tablekey, otherkey]),
           newname = lambda tablename, othername: '/'.join([tablename, othername]),
           newvariable = lambda tablevar, othervar, args, kwargs: tablevar.operation(othervar, *args, method='divide', **kwargs))
def divide(dataarray, otherdataarray, *args, **kwargs):
    assert all([dataarray.attrs[key] == otherdataarray.attrs[key] for key in dataarray.attrs.keys() if key in otherdataarray.attrs.keys()])         
    newdataarray = dataarray / otherdataarray
    scope = dataarray.attrs
    scope.update(otherdataarray.attrs)
    newdataarray.attrs = scope
    newdataarray.name = dataarray.name
    return newdataarray


@operation
def combine(dataarray, otherdataarray, *args, onscope, **kwargs):
    newdataxarray = xr.concat([dataarray, otherdataarray], pd.Index([dataarray.attrs[onscope], otherdataarray.attrs[onscope]], name=onscope))
    newdataxarray.name = dataarray.name
    return newdataxarray


@operation
def merge(dataarray, otherdataarray, *args, onaxis, **kwargs):
    newdataxarray = xr.concat([dataarray, otherdataarray], dim=onaxis)
    newdataxarray.name = dataarray.name
    return newdataxarray


@operation
def append(dataarray, otherdataarray, *args, toaxis, **kwargs):
    otherdataarray = otherdataarray.expand_dims(toaxis)
    otherdataarray.coords[toaxis] = pd.Index([otherdataarray.attrs.pop(toaxis)], name=toaxis)
    newdataxarray = xr.concat([dataarray, otherdataarray], dim=toaxis)
    newdataxarray.name = dataarray.name
    return newdataxarray







