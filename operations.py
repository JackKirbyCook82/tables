# -*- coding: utf-8 -*-
"""
Created on Sun Jun 2 2019
@name    Operation Functions
@author: Jack Kriby Cook

"""

from functools import update_wrapper
import xarray as xr
import pandas as pd

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['add', 'subtract', 'multiply', 'divide', 'concat', 'merge', 'append', 'layer']
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""


_ALLCHAR = '*'
_aslist = lambda items: [items] if not isinstance(items, (list, tuple)) else list(items)     


def operation_loop(function):
    def wrapper(table, others, *args, **kwargs):
        newtable = table
        for other in _aslist(others): newtable = function(newtable, other, *args, **kwargs)
        return newtable
    update_wrapper(wrapper, function)
    return wrapper


def operation(name_function = lambda name, other, args, kwargs: kwargs.get('name', name), 
              datakey_function = lambda datakey, other, args, kwargs: datakey, 
              variable_function = lambda variable, other, args, kwargs: None):
    def decorator(dataarray_function):
        def wrapper(table, other, *args, **kwargs):
            TableClass = table.__class__
            assert table.layers == other.layers == 1

            name, datakey = table.name, table.datakeys[0]
            othername, otherdatakey = other.name, other.datakeys[0]

            dataarray, variables, scope = table.dataarrays[datakey], table.variables, table.scope
            otherdataarray, othervariables, otherscope = other.dataarrays[otherdatakey], other.variables, table.scope        
            dataarray.attrs, otherdataarray.attrs = scope, otherscope
                      
            newdatakey = datakey_function(datakey, otherdatakey, args, kwargs)
            newname = name_function(name, othername, args, kwargs)                      
            newvariable = variable_function(variables[datakey], othervariables[otherdatakey], args, kwargs)
            
            newdataarray, newscope = dataarray_function(dataarray, otherdataarray, *args, variables=variables, othervariables=othervariables, **kwargs)  
            
            #newheaderkeys = tuple(newdataarray.dims)
            #newscopekeys = tuple(newscope.keys())
            
            #newvariables = {newdatakey:newvariable}
            #newvariables.update(_getvariables(variables, othervariables, newheaderkeys))
            #newvariables.update(_getvariables(variables, othervariables, newscopekeys))
            
            newdataset = newdataarray.to_dataset(name=newdatakey) 
            newdataset.attrs = newscope            
            return TableClass(data=newdataset, variables=newvariables, name=newname)
            
        update_wrapper(wrapper, dataarray_function)
        return wrapper
    return decorator


@operation()
def add(dataarray, other, *args, onscope, variables, othervariables, **kwargs):
    attrs, otherattrs = dataarray.attrs, other.attrs
    assert dataarray.dims == other.dims
    assert dataarray.shape == other.shape    
    assert dataarray.name == other.name

    newdataarray = dataarray + other
    #newattr = str(variables.fromstr(attrs[onscope]).add(othervariables.fromstr(otherattrs[onscope]), *args, **kwargs))
    #newattrs = {key:value if key != onscope else newattr for key, value in attrs.items()}
    return newdataarray, newattrs


@operation()
def subtract(dataarray, other, *args, onscope, variables, othervariables, **kwargs):
    attrs, otherattrs = dataarray.attrs, other.attrs
    assert dataarray.dims == other.dims
    assert dataarray.shape == other.shape    
    assert dataarray.name == other.name

    newdataarray = dataarray - other
    #newattr = str(variables.fromstr(attrs[onscope]).subtract(othervariables.fromstr(otherattrs[onscope]), *args, **kwargs))
    #newattrs = {key:value if key != onscope else newattr for key, value in attrs.items()}
    return newdataarray, newattrs
    
      
@operation(name_function = lambda name, other, args, kwargs: kwargs.get('name', '*'.join([name, other])),
           datakey_function = lambda datakey, other, args, kwargs: '*'.join([datakey, other]),
           variable_function = lambda variable, other, args, kwargs: variable.operation(other, *args, method='multiply', **kwargs))
def multiply(dataarray, other, *args, variables, othervariables, **kwargs):
    attrs, otherattrs = dataarray.attrs, other.attrs
    assert dataarray.dims == other.dims
    assert dataarray.shape == other.shape
    
    newdataarray = dataarray * other
    #newattrs = {key:attrs[key] if key in otherattrs.keys() else other[key] for key in list(set(*attrs.keys(), *otherattrs.keys()))}
    return newdataarray, newattrs
    

@operation(name_function = lambda name, other, args, kwargs: kwargs.get('name', '/'.join([name, other])),
           datakey_function = lambda datakey, other, args, kwargs: '/'.join([datakey, other]),
           variable_function = lambda variable, other, args, kwargs: variable.operation(other, *args, method='divide', **kwargs))
def divide(dataarray, other, *args, variables, othervariables, **kwargs):
    attrs, otherattrs = dataarray.attrs, other.attrs
    assert dataarray.dims == other.dims
    assert dataarray.shape == other.shape
    
    newdataarray = dataarray / other
    #newattrs = {key:attrs[key] if key in attrs.keys() else otherattrs[key] for key in list(set([*attrs.keys(), *otherattrs.keys()]))}
    return newdataarray, newattrs


@operation()
def concat(dataarray, other, *args, onaxis, variables, othervariables, **kwargs):
    attrs, otherattrs = dataarray.attrs, other.attrs
    assert dataarray.name == other.name
    assert dataarray.dims == other.dims
    #assert attrs == otherattrs
   
    newdataarray = xr.concat([dataarray, other], dim=onaxis)
    #newattrs = {key:value for key, value in attrs.items()}
    return newdataarray, newattrs


@operation()
def merge(dataarray, other, *args, onscope, variables, othervariables,  **kwargs):
    attrs, otherattrs = dataarray.attrs, other.attrs
    assert dataarray.name == other.name
    assert dataarray.dims == other.dims
    assert dataarray.shape == other.shape 
    #assert attrs.keys() == otherattrs.keys()
    #coreattrs = {key:value for key, value in attrs.items() if value == otherattrs[key]}
    #assert len(attrs) == len(otherattrs) == len(coreattrs) - 1 

    newdataarray = xr.concat([dataarray, other], pd.Index([attrs[onscope], otherattrs[onscope]], name=onscope))
    #newattrs = {key:value for key, value in attrs.items() if key != onscope}
    return newdataarray, newattrs


@operation()
def append(dataarray, other, *args, toaxis, variables, othervariables, **kwargs):
    attrs, otherattrs = dataarray.attrs, other.attrs
    assert dataarray.name == other.name
    assert dataarray.dims == other.dims + 1
    #coreattrs = {key:attrs[key] for key in set([*attrs.keys(), *otherattrs.keys()]) if attrs[key] == otherattrs[key]}
    #assert len(attrs) == len(otherattrs) - 1 == len(coreattrs)
    
    other = other.expand_dims(pd.Index([otherattrs[toaxis]], name=toaxis))
    newdataarray = xr.concat([dataarray, other], dim=toaxis)
    #newattrs = {key:value for key, value in attrs.items()}
    return newdataarray, newattrs
    

@operation_loop
def layer(table, other, *args, name, **kwargs):
    TableClass = table.__class__
    assert table.dim == other.dim
    assert table.shape == other.shape
    print(table, other)
    datakeys, headerkeys, scopekeys = table.datakeys, table.headerkeys, table.scopekeys
    otherdatakeys, otherheaderkeys, otherscopekeys = other.datakeys, other.headerkeys, other.scopekeys
        
    assert all([datakey not in otherdatakeys for datakey in datakeys]) 
    assert headerkeys == otherheaderkeys
   
    dataset, variables, scope = table.dataset, table.variables, table.scope
    otherdataset, othervariables, otherscope = other.dataset, other.variables, other.scope      

    #newscope = {key:scope[key] if key in scope.keys() else otherscope[key] for key in list(set([*scopekeys, *otherscopekeys]))}    
    #newscope = {key:value for key, value in newscope.items() if key not in (*datakeys, *otherdatakeys, *headerkeys)}
    
    #newvariables = _getvariables(variables, othervariables, (*datakeys, *otherdatakeys))
    #newvariables.update(_getvariables(variables, othervariables, headerkeys))
    #newvariables.update(_getvariables(variables, othervariables, newscope.keys()))

    newdataset = xr.merge([dataset, otherdataset])  
    newdataset.attrs = newscope
    
    return TableClass(data=newdataset, variables=newvariables, name=name)  









