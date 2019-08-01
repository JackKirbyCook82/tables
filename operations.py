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
__all__ = ['multiply', 'divide', 'layer']
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""


_aslist = lambda items: [items] if not isinstance(items, (list, tuple)) else list(items)


#def internal_operation(newdatakey_function, newvariable_function):
#    def decorator(function):
#        def wrapper(table, *args, datakey, otherdatakey, **kwargs):
#            TableClass = table.__class__       
#            dataset, dataarrays, variables, name = table.dataset, table.dataarrays, table.variables.copy(), table.name
#
#            newdatakey = newdatakey_function(datakey, otherdatakey)
#            newvariable = newvariable_function(variables[datakey], variables[otherdatakey], args, kwargs)
#            newdataarray = function(dataarrays[datakey], dataarrays[otherdatakey], *args, **kwargs)
#            
#            newdataset = xr.merge([dataset, newdataarray.to_dataset(name=newdatakey)])   
#            variables.update({newdatakey:newvariable})
#            newdataset.attrs = dataset.attrs          
#            return TableClass(data=newdataset, variables=variables, name=kwargs.get('name', name))
#      
#        update_wrapper(wrapper, function)
#        return wrapper
#    return decorator    


#def external_operation(function):
#    def wrapper(table, other, *args, **kwargs):
#        TableClass = table.__class__
#        assert table.datakeys == other.datakeys
#        assert table.variables == other.variables
#        
#        newdataarrays = {datakey:function(dataarray, otherdataarray, *args, **kwargs) for datakey, dataarray, otherdataarray in zip(table.datakeys, table.dataarrays, other.dataarrays)}
#        newdataset = xr.merge([newdataarray.to_dataset(name=datakey) for datakey, newdataarray in newdataarrays.items()])          
#        return TableClass(data=newdataset, variables=table.variables, name=kwargs.get('name', table.name))
#    
#    update_wrapper(wrapper, function)
#    return wrapper


def operation(newname_function, newdatakey_function, newvariable_function):
    def decorator(function):
        def wrapper(table, other, *args, **kwargs):
            TableClass = table.__class__
            assert table.layers == 1 and other.layers == 1
            assert table.dim == other.dim
            assert table.shape == other.shape
            assert table.headerkeys == other.headerkeys            

            name, datakey, headerkeys = table.name, table.datakeys[0], table.headerkeys
            othername, otherdatakey, otherheaderkeys = other.name, other.datakeys[0], other.headerkeys
            
            scope, variables =  table.scope.copy(), table.variables.copy()
            otherscope, othervariables = other.scope.copy(), other.variables.copy()

            for key, value in scope.items():
                if key == otherdatakey: pass
                if key in otherheaderkeys: pass
                    
            for key, value in otherscope.items():
                if key == datakey: pass
                if key in headerkeys: pass
                    
            dataarray, variable = table.dataarrays[datakey].copy(), table.variables[datakey]
            otherdataarray, othervariable = other.dataarrays[otherdatakey].copy(), other.variables[otherdatakey]
            
            newname = kwargs.get('name', newname_function(name, othername))
            newdatakey = newdatakey_function(datakey, otherdatakey)
            newheaderkeys = table.headerkeys
            newscopekeys = tuple([scopekey for scopekey in set([*table.scopekeys, *other.scopekeys]) if scopekey not in (newdatakey, *newheaderkeys)])

            newdataarray = function(dataarray, otherdataarray, *args, **kwargs)
            newvariable = newvariable_function(variable, othervariable, args, kwargs)
            #newscope = {key:scope.get(key, otherscope[key]) for key in newscopekeys}            
            newscope = {}   
            ###
            
            variablefunction = lambda keys: {key:(variables[key] if key in variables.keys() else othervariables[key]) for key in keys}   
            newvariables = variablefunction(newscopekeys)
            newvariables.update(variablefunction(newheaderkeys))
            newvariables.update({newdatakey:newvariable})
            
            newdataset = newdataarray.to_dataset(name=newdatakey)  
            newdataset.attrs = newscope   
            return TableClass(data=newdataset, variables=newvariables, name=newname)
        update_wrapper(wrapper, function)
        return wrapper
    return decorator


@operation(newname_function = lambda name, other: '*'.join([name, other]),
           newdatakey_function = lambda datakey, other: '*'.join([datakey, other]),
           newvariable_function = lambda variable, other, args, kwargs: variable.operation(other, *args, method='multiply', **kwargs))
def multiply(dataarray, otherdataarray, *args, **kwargs):     
    return dataarray * otherdataarray


@operation(newname_function = lambda name, other: '*'.join([name, other]),
           newdatakey_function = lambda datakey, other: '*'.join([datakey, other]),
           newvariable_function = lambda variable, other, args, kwargs: variable.operation(other, *args, method='multiply', **kwargs))
def divide(dataarray, otherdataarray, *args, **kwargs):     
    return dataarray / otherdataarray


def layer(table, other, *args, name, **kwargs):
    TableClass = table.__class__
    assert table.dim == other.dim
    assert table.shape == other.shape
    
    dataset, variables, scope = table.dataset.copy(), table.variables.copy(), table.scope.copy()
    otherdataset, othervariables, otherscope = other.dataset.copy(), other.variables.copy(), other.scope.copy()
    
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
    
    #newscope = {key:scope.get(key, otherscope[key]) for key in newscopekeys}
    newscope = {}
    ###
    
    newdataset = xr.merge([dataset, otherdataset])  
    newdataset.attrs = newscope
    return TableClass(data=newdataset, variables=newvariables, name=name)    


#def combine(dataarray, otherdataarray, *args, onscope, **kwargs):
#    newdataarray = xr.concat([dataarray, otherdataarray], pd.Index([dataarray.attrs[onscope], otherdataarray.attrs[onscope]], name=onscope))
#    newdataarray.name = dataarray.name
#    return newdataarray


#def merge(dataarray, otherdataarray, *args, onaxis, **kwargs):
#    newdataarray = xr.concat([dataarray, otherdataarray], dim=onaxis)
#    newdataarray.name = dataarray.name
#    return newdataarray


#def append(dataarray, otherdataarray, *args, toaxis, **kwargs):
#    otherdataarray = otherdataarray.expand_dims(toaxis)
#    otherdataarray.coords[toaxis] = pd.Index([otherdataarray.attrs.pop(toaxis)], name=toaxis)
#    newdataarray = xr.concat([dataarray, otherdataarray], dim=toaxis)
#    newdataarray.name = dataarray.name
#    return newdataarray



















