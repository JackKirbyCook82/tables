# -*- coding: utf-8 -*-
"""
Created on Sun Jun 2 2019
@name    Operation Functions
@author: Jack Kriby Cook

"""

from functools import update_wrapper
import xarray as xr

import utilities.xarrays as xar

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['add', 'subtract', 'multiply', 'divide', 'concat', 'merge', 'append', 'layer']
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""


_aslist = lambda items: [items] if not isinstance(items, (list, tuple)) else list(items)     


def operation(name_function = lambda name, other: name, datakey_function = lambda datakey, other: datakey):
    def decorator(dataarray_function):
        def wrapper(table, other, *args, axis=None, **kwargs):
            assert isinstance(other, type(table))    
            TableClass = table.__class__
            assert table.layers == other.layers == 1
            assert table.dim == other.dim
            assert table.shape == other.shape  
   
            method = dataarray_function.__name__
            if axis is None: axistype = 'data'
            elif axis in table.headerkeys: axistype = 'header'
            elif axis in table.scopekeys: axistype = 'scope'
            else: raise ValueError(axis)    
    
            name, othername = table.name, other.name
            datakey, otherdatakey = table.datakeys[0], other.datakeys[0]   
            if not axis: axis, otheraxis = datakey, otherdatakey  
            else: axis, otheraxis = axis, axis
            dataarray, otherdataarray = table.dataarrays[datakey], other.dataarrays[otherdatakey]  
            scope, otherscope = table.scope, other.scope
            variables, othervariables = table.variables, other.variables

            if axistype != 'data': assert table.datakeys == other.datakeys
            assert table.headerkeys == other.headerkeys
            for key in set([*table.headerkeys, *other.headerkeys]):
                if all([key != axis, key != otheraxis]): 
                    assert all([hdritem == otheritem for hdritem, otheritem in zip(table.headers[key], other.headers[key])])
            for key in set([*table.scopekeys, *other.scopekeys]):        
                if all([key != axis, key != otheraxis, key in table.scopekeys, key in other.scopekeys]):
                    assert table.scope[key] == other.scope[key]
            
            axes_function = lambda item, otheritem: getattr(variables[axis].fromstr(item), method)(othervariables[otheraxis].fromstr(otheritem), *args, **kwargs)           
 
            newname = kwargs.get('name', name_function(name, othername))
            newdatakey = datakey_function(datakey, otherdatakey)
            newdataarray = dataarray_function(dataarray, otherdataarray, *args, **kwargs)
            newvariable = variables[axis].operation(othervariables[otheraxis], *args, method=method, **kwargs) 
            if axistype == 'data': newaxes = None
            elif axistype == 'header': newaxes = [axes_function(item, otheritem) for item, otheritem in zip(table.headers[axis], other.headers[otheraxis])]  
            elif axistype == 'scope': newaxes = axes_function(table.scope[axis], other.scole[otheraxis])                   
            else: return ValueError(axistype)

            newvariables = {key:value for key, value in variables.items()}
            newvariables.update({newdatakey:newvariable})
            newattrs = {key:scope[key] for key in set([*scope.keys(), *otherscope.keys()]) if scope.get(key, None) == otherscope.get(key, None)}
            newdataset = newdataarray.to_dataset(name=newdatakey) 
            newdataset.attrs = newattrs
            if axistype == 'data': pass
            elif axistype == 'header': newdataset = newdataset.assign.coords(**{axis:newaxes})
            elif axistype == 'scope': newdataset.attrs[axis] == newaxes                  
            else: return ValueError(axistype)
            return TableClass(data=newdataset, variables=newvariables, name=newname)        
        update_wrapper(wrapper, dataarray_function)
        return wrapper   
    return decorator     


def operationloop(function):
    def wrapper(table, others, *args, **kwargs):
        newtable = table
        for other in _aslist(others): newtable = function(newtable, other, *args, **kwargs)
        return newtable
    update_wrapper(wrapper, function)
    return wrapper


@operation()
def add(dataarray, other, *args, **kwargs): return dataarray + other  

@operation()
def subtract(dataarray, other, *args, **kwargs): return dataarray - other  

@operation(name_function = lambda name, other: '*'.join([name, other]), 
           datakey_function = lambda datakey, other: '*'.join([datakey, other]))
def multiply(dataarray, other, *args, **kwargs): return dataarray * other

@operation(name_function = lambda name, other: '/'.join([name, other]), 
           datakey_function = lambda datakey, other: '/'.join([datakey, other]))
def divide(dataarray, other, *args, **kwargs): return dataarray / other


def concat(table, other, *args, axis, **kwargs):
    TableClass = table.__class__
    assert isinstance(other, type(table))    
    assert table.layers == other.layers == 1
    assert table.dims == other.dims
    assert table.shape == other.shape 
    assert table.variables == other.variables    

    datakey, otherdatakey = table.datakeys[0], other.datakeys[0]
    assert datakey == otherdatakey
    newdataarray = xar.xarray_concat(table.dataarrays[datakey], other.dataarrays[otherdatakey], *args, onaxis=axis, **kwargs)
    newdataset = newdataarray.to_dataset(name=datakey) 
    newdataset.attrs = newdataarray.attrs      
    return TableClass(data=newdataset, variables=table.variables, name=table.name)


def merge(table, other, *args, axis, **kwargs):
    table, other = table.expand(axis), other.expand(axis)
    return concat(table, other, *args, onaxis=axis, **kwargs)


def append(table, other, *args, axis, **kwargs):
    other = other.expand(axis)
    return concat(table, other, *args, onaxis=axis, **kwargs)


@operationloop
def layer(table, other, *args, name, **kwargs):
    assert isinstance(other, type(table))    
    TableClass = table.__class__
    assert other.layers == 1
    assert table.dim == other.dim
    assert table.shape == other.shape

    datakeys, otherdatakey = table.datakeys, other.datakeys[0]
    dataset, otherdataset = table.dataset, other.dataset,
    scope, otherscope = table.scope, other.scope   
    variables, othervariables = table.variables, other.variables    

    assert otherdatakey not in datakeys
    assert table.headerkeys == other.headerkeys
    for key in set([*table.headerkeys, *other.headerkeys]):
        assert all([hdritem == otheritem for hdritem, otheritem in zip(table.headers[key], other.headers[key])])
    for key in set([*table.scopekeys, *other.scopekeys]):        
        if all([key in table.scopekeys, key in other.scopekeys]):
            assert table.scope[key] == other.scope[key]
    for key in set([*table.headerkeys, *other.headerkeys]):
        assert variables[key] == othervariables[key]
    for key in set([*table.scopekeys, *other.scopekeys]):
        if all([key in table.scopekeys, key in other.scopekeys]):
            assert variables[key] == othervariables[key]
            
    variable_function = lambda key: variables[key] if key in variables.keys() else othervariables[key]
        
    newdataset = xr.merge([dataset, otherdataset])  
    newattrs = {key:scope[key] for key in set([*scope.keys(), *otherscope.keys()]) if scope.get(key, None) == otherscope.get(key, None)}    
    newdataset.attrs = newattrs
    newvariables = {key:variable_function(key) for key in (*datakeys, otherdatakey, *set([*table.headerkeys, *other.headerkeys]))}
    newvariables.update({key:variable_function(key) for key in newattrs.keys()})
    return TableClass(data=newdataset, variables=newvariables, name=name)  









