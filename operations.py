# -*- coding: utf-8 -*-
"""
Created on Sun Jun 2 2019
@name    Operation Functions
@author: Jack Kriby Cook

"""

from functools import update_wrapper
from itertools import chain
import pandas as pd
import xarray as xr

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['multiply', 'divide', 'combine', 'merge', 'append', 'layer']
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""


_aslist = lambda items: [items] if not isinstance(items, (list, tuple)) else list(items)


def internal_operation(newdatakey_function, newvariable_function):
    def decorator(function):
        def wrapper(table, *args, datakey, otherdatakey, **kwargs):
            TableClass = table.__class__       
            dataset, dataarrays, variables = table.dataset, table.dataarrays, table.variables.copy()

            newdatakey = newdatakey_function(datakey, otherdatakey)
            newvariable = newvariable_function(variables[datakey], variables[otherdatakey], args, kwargs)
            newdataarray = function(dataarrays[datakey], dataarrays[otherdatakey], *args, **kwargs)
            
            newdataset = xr.merge([dataset, newdataarray.to_dataset(name=newdatakey)])   
            variables.update({newdatakey:newvariable})
            newdataset.attrs = dataset.attrs          
            return TableClass(data=newdataset, variables=variables)
      
        update_wrapper(wrapper, function)
        return wrapper
    return decorator    


def external_operation(function):
    def wrapper(table, other, *args, **kwargs):
        TableClass = table.__class__
        assert table.datakeys == other.datakeys
        assert table.variables == other.variables
        
        newdataarrays = {datakey:function(dataarray, otherdataarray, *args, **kwargs) for datakey, dataarray, otherdataarray in zip(table.datakeys, table.dataarrays, other.dataarrays)}
        newdataset = xr.merge([newdataarray.to_dataset(name=datakey) for datakey, newdataarray in newdataarrays.items()])          
        return TableClass(data=newdataset, variables=table.variables)
    
    update_wrapper(wrapper, function)
    return wrapper


@internal_operation(newdatakey_function = lambda tablekey, otherkey: '*'.join([tablekey, otherkey]),
                    newvariable_function = lambda tablevar, othervar, args, kwargs: tablevar.operation(othervar, *args, method='multiply', **kwargs))
def multiply(dataarray, otherdataarray, *args, **kwargs): 
    assert all([dataarray.attrs[key] == otherdataarray.attrs[key] for key in dataarray.attrs.keys() if key in otherdataarray.attrs.keys()])       
    newdataarray = dataarray * otherdataarray
    scope = dataarray.attrs.copy()
    scope.update(otherdataarray.attrs)
    newdataarray.attrs = scope
    return newdataarray


@internal_operation(newdatakey_function = lambda tablekey, otherkey: '/'.join([tablekey, otherkey]),
                    newvariable_function = lambda tablevar, othervar, args, kwargs: tablevar.operation(othervar, *args, method='divide', **kwargs))
def divide(dataarray, otherdataarray, *args, **kwargs):
    assert all([dataarray.attrs[key] == otherdataarray.attrs[key] for key in dataarray.attrs.keys() if key in otherdataarray.attrs.keys()])         
    newdataarray = dataarray / otherdataarray
    scope = dataarray.attrs.copy()
    scope.update(otherdataarray.attrs)
    newdataarray.attrs = scope
    return newdataarray


@external_operation
def combine(dataarray, otherdataarray, *args, onscope, **kwargs):
    newdataarray = xr.concat([dataarray, otherdataarray], pd.Index([dataarray.attrs[onscope], otherdataarray.attrs[onscope]], name=onscope))
    newdataarray.name = dataarray.name
    return newdataarray


@external_operation
def merge(dataarray, otherdataarray, *args, onaxis, **kwargs):
    newdataarray = xr.concat([dataarray, otherdataarray], dim=onaxis)
    newdataarray.name = dataarray.name
    return newdataarray


@external_operation
def append(dataarray, otherdataarray, *args, toaxis, **kwargs):
    otherdataarray = otherdataarray.expand_dims(toaxis)
    otherdataarray.coords[toaxis] = pd.Index([otherdataarray.attrs.pop(toaxis)], name=toaxis)
    newdataarray = xr.concat([dataarray, otherdataarray], dim=toaxis)
    newdataarray.name = dataarray.name
    return newdataarray


def layer(table, other, *args, **kwargs):
    TableClass = table.__class__
    dataset, variables, attrs = table.dataset, table.variables, table.dataset.attrs.copy()
    otherdataset, othervariables, otherattrs = other.dataset, other.variables, other.dataset.attrs
    
    assert all([datakey not in other.datakeys for datakey in table.datakeys])
    assert table.headerkeys == other.headerkeys
    newdatakeys = (*table.datakeys, *other.datakeys)    
    newheaderkeys = table.headerkeys
    newscopekeys = tuple([scopekey for scopekey in set([*table.scopekeys, *other.scopekeys]) if scopekey not in (*newdatakeys, *newheaderkeys)])
    
    variablefunction = lambda keys: {key:(variables[key] if key in variables.keys() else othervariables[key]) for key in keys}
    newvariables = variablefunction(newscopekeys)
    newvariables.update(variablefunction(newheaderkeys))
    newvariables.update({datakey:variables[datakey] for datakey in table.datakeys})
    newvariables.update({datakey:othervariables[datakey] for datakey in other.datakeys})
    
    attrs.update(otherattrs)   
    newdataset = xr.merge([dataset, otherdataset])  
    newdataset.attrs = {scopekey:scopevalue for scopekey, scopevalue in attrs.items() if scopekey not in newdatakeys}
    return TableClass(data=newdataset, variables=newvariables)    
















