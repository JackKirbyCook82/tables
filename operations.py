# -*- coding: utf-8 -*-
"""
Created on Sun Jun 2 2019
@name    Operation Functions
@author: Jack Kriby Cook

"""

from functools import update_wrapper
import xarray as xr
import pandas as pd

import utilities.xarrays as xar

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['add', 'subtract', 'multiply', 'divide', 'concat', 'merge', 'append', 'layer']
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""


_aslist = lambda items: [items] if not isinstance(items, (list, tuple)) else list(items)     


#def operation_loop(function):
#    def wrapper(table, others, *args, **kwargs):
#        newtable = table
#        for other in _aslist(others): newtable = function(newtable, other, *args, **kwargs)
#        return newtable
#    update_wrapper(wrapper, function)
#    return wrapper


def operation():
    def decorator(function):
        def wrapper(table, other, *args, **kwargs):
            assert isinstance(other, type(table))    
            TableClass = table.__class__
            assert table.layers == other.layers == 1
            assert table.dims == other.dims
            assert table.shape == other.shape  
            newdataset, newvariables, newname = function(table, other, *args, **kwargs)
            return TableClass(data=newdataset, variables=newvariables, name=newname)           
        update_wrapper(wrapper, function)
        return wrapper
    return decorator
    
        
def data_operation(namefunction, datakeyfunction):
    def decorator(function):
        @operation()
        def wrapper(table, other, *args, **kwargs):
            
            newname = kwargs.get('name', namefunction(table.name, other.name))
            newdatakey = datakeyfunction(table.name, other.name)
            
            return newdataset, newvariables, newname
        return wrapper
        update_wrapper(wrapper, function)
    return decorator            


def scope_operation():
    def decorator(function):
        @operation
        def wrapper(table, other, *args, **kwargs):
            
            newname = kwargs.get('name', table.name)
            newdatakey = table.datakey
            
            return newdataset, newvariables, newname
        return wrapper
        update_wrapper(wrapper, function)
    return decorator


#def operation(core):
#    def decorator(function):
#        def wrapper(table, other, *args, **kwargs):
#            TableClass = table.__class__
#            assert isinstance(other, type(table))    
#            assert table.layers == other.layers == 1
#            assert table.dims == other.dims
#            assert table.shape == other.shape          
#            
#            table, other, corekey = aligntables(table, other, *args, core, **kwargs)
#            
#            assert table.axeskeys == other.axeskeys
#            for axeskey in table.axeskeys:
#                if axeskey != corekey: assert table.axes[corekey] == other.axes[corekey]   
#                assert table.variables[axeskey] == other.variables[axeskey]
#                           
#            return TableClass(data=newdataset, variables=newvariables, name=newname)
#            
#        update_wrapper(wrapper, function)
#        return wrapper
#    return decorator


@operation(core='onscope')
def add(table, other, *args, onscope, **kwargs):
    assert table.datakeys == other.datakeys 

    dataarrays, otherdataarrays = table.dataarrays, other.dataarrays
    scope, otherscope = table.scope, other.scope
    variables, othervariables = table.varaiables, other.variables
    
    datakey, otherdatakey = table.datakeys[0], other.datakeys[0]
    
    newname = kwargs.get('name', table.name)
    newdataarray = dataarrays[datakey] + otherdataarrays[otherdatakey]
    newattr = str(variables[onscope].fromstr(scope[onscope]).add(othervariables[onscope].fromstr(otherscope[onscope]), *args, **kwargs))
    newattrs = {key:value if key != onscope else newattr for key, value in scope.items()}
    newdataset = newdataarray.to_dataset(name=datakey) 
    newdataset.attrs = newattrs
    return    
    

@operation(core='onscope')
def subtract(table, other, *args, onscope, **kwargs):
    assert table.datakeys == other.datakeys 

    dataarrays, otherdataarrays = table.dataarrays, other.dataarrays
    scope, otherscope = table.scope, other.scope
    variables, othervariables = table.varaiables, other.variables
    
    datakey, otherdatakey = table.datakeys[0], other.datakeys[0]
    
    newname = kwargs.get('name', table.name)
    newdataarray = dataarrays[datakey] - otherdataarrays[otherdatakey]
    newattr = str(variables[onscope].fromstr(scope[onscope]).subtract(othervariables[onscope].fromstr(otherscope[onscope]), *args, **kwargs))
    newattrs = {key:value if key != onscope else newattr for key, value in scope.items()}
    newdataset = newdataarray.to_dataset(name=datakey) 
    newdataset.attrs = newattrs
    return 


@operation(core='ondata')
def multiply(table, other, *args, **kwargs):
    dataarrays, otherdataarrays = table.dataarrays, other.dataarrays
    scope, otherscope = table.scope, other.scope
    variables, othervariables = table.varaiables, other.variables
    
    datakey, otherdatakey = table.datakeys[0], other.datakeys[0]
    
    newname = kwargs.get('name', '*'.join([table.name, other.name]))
    newdatakey = '*'.join([datakey, other])
    newdataarray = dataarrays[datakey] * otherdataarrays[otherdatakey]
    newattrs = scope
    newdataset = newdataarray.to_dataset(name=newdatakey) 
    newdataset.attrs = newattrs
    newvariable = variables[datakey].operation(othervariables[datakey], *args, method='multiply', **kwargs)
    newvariables = variables.copy()
    newvariables.update({newdatakey:newvariable})
    return     
    

@operation(core='ondata')
def divide(table, other, *args, **kwargs):
    dataarrays, otherdataarrays = table.dataarrays, other.dataarrays
    scope, otherscope = table.scope, other.scope
    variables, othervariables = table.varaiables, other.variables
    
    datakey, otherdatakey = table.datakeys[0], other.datakeys[0]
    
    newname = kwargs.get('name', '*'.join([table.name, other.name]))
    newdatakey = '/'.join([datakey, other])    
    newdataarray = dataarrays[datakey] * otherdataarrays[otherdatakey]
    newattrs = scope
    newdataset = newdataarray.to_dataset(name=newdatakey) 
    newdataset.attrs = newattrs
    newvariable = variables[datakey].operation(othervariables[datakey], *args, method='divide', **kwargs)
    newvariables = variables.copy()
    newvariables.update({newdatakey:newvariable})
    return  


def concat(table, other, *args, onaxis, **kwargs):
    TableClass = table.__class__
    assert isinstance(other, type(table))    
    assert table.layers == other.layers == 1
    assert table.dims == other.dims
    assert table.shape == other.shape 
    assert table.variables == other.variables    

    datakey, otherdatakey = table.datakeys[0], other.datakeys[0]
    assert datakey == otherdatakey
    newdataarray = xar.xarray_concat(table.dataarrays[datakey], other.dataarrays[otherdatakey], *args, onaxis=onaxis, **kwargs)
    newdataset = newdataarray.to_dataset(name=datakey) 
    newdataset.attrs = newdataarray.attrs      
    return TableClass(data=newdataset, variables=table.variables, name=table.name)


def merge(table, other, *args, onscope, **kwargs):
    table, other = table.expand(onscope), other.expand(onscope)
    return concat(table, other, *args, onaxis=onscope, **kwargs)


def append(table, other, *args, toaxis, **kwargs):
    other = other.expand(toaxis)
    return concat(table, other, *args, onaxis=toaxis, **kwargs)


#@operation_loop
#def layer(table, other, *args, name, **kwargs):
#    TableClass = table.__class__
#    assert table.dim == other.dim
#    assert table.shape == other.shape
#    print(table, other)
#    datakeys, headerkeys, scopekeys = table.datakeys, table.headerkeys, table.scopekeys
#    otherdatakeys, otherheaderkeys, otherscopekeys = other.datakeys, other.headerkeys, other.scopekeys
#        
#    assert all([datakey not in otherdatakeys for datakey in datakeys]) 
#    assert headerkeys == otherheaderkeys
#   
#    dataset, variables, scope = table.dataset, table.variables, table.scope
#    otherdataset, othervariables, otherscope = other.dataset, other.variables, other.scope      
#
#    #newscope = {key:scope[key] if key in scope.keys() else otherscope[key] for key in list(set([*scopekeys, *otherscopekeys]))}    
#    #newscope = {key:value for key, value in newscope.items() if key not in (*datakeys, *otherdatakeys, *headerkeys)}
#    
#    #newvariables = _getvariables(variables, othervariables, (*datakeys, *otherdatakeys))
#    #newvariables.update(_getvariables(variables, othervariables, headerkeys))
#    #newvariables.update(_getvariables(variables, othervariables, newscope.keys()))
#
#    newdataset = xr.merge([dataset, otherdataset])  
#    newdataset.attrs = newscope
#    
#    return TableClass(data=newdataset, variables=newvariables, name=name)  









