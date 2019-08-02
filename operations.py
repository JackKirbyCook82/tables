# -*- coding: utf-8 -*-
"""
Created on Sun Jun 2 2019
@name    Operation Functions
@author: Jack Kriby Cook

"""

from functools import update_wrapper
import xarray as xr
from collections import OrderedDict as ODict

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['layer', 'add', 'subtract', 'multiply', 'divide', 'concat', 'merge', 'append']
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""


_aslist = lambda items: [items] if not isinstance(items, (list, tuple)) else list(items)     


def operation():
    def decorator(function):
        def wrapper(table, other, *args, **kwargs):
            pass
        update_wrapper(wrapper, function)
        return wrapper
    return decorator


@operation()
def add(dataarray, other, *args, **kwargs):
    pass


@operation()
def subtract(dataarray, other, *args, **kwargs):
    pass


@operation()
def multiply(dataarray, other, *args, **kwargs):
    pass


@operation()
def divide(dataarray, other, *args, **kwargs):
    pass


@operation()
def concat(dataarray, other, *args, **kwargs):
    pass


@operation()
def merge(dataarray, other, *args, **kwargs):
    pass


@operation()
def append(dataarray, other, *args, **kwargs):
    pass


def layer(table, other, *args, **kwargs):
    pass










#def update_variables(variables, others, newvariables, *args, keys, **kwargs):
#    updated = {}
#    for key in keys:
#        try: updated[key] = newvariables[key]
#        except KeyError: 
#            if key in variables.keys() and key in others.keys(): 
#                assert variables[key] == others[key]
#                updated[key] = variables[key]
#            else: updated[key] = variables[key] if key in variables.keys() else others[key]
#    return updated


#def update_scope(scope, other, *args, **kwargs):
#    scopekeys = list(*set([*scope.keys(), *other.keys()]))
#    updated = {}
#    for key in scopekeys:
#        if key in scope.keys() and key in other.keys():
#            assert scope[key] == other[key]
#            updated[key] = scope[key]
#        else: updated[key] = scope[key] if key in scope.keys() else other[key]
#    return updated


#def operation_loop(function):
#    def wrapper(table, others, *args, **kwargs):
#        newtable = table.copy()
#        for other in _aslist(others): newtable = function(newtable, other, *args, **kwargs)
#        return newtable
#    update_wrapper(wrapper, function)
#    return wrapper


#def dataarray_operation(name_function, datakey_function, variable_function):
#    def decorator(dataarray_function): 
#        def wrapper(table, other, *args, **kwargs):
#            TableClass = table.__class__
#            assert table.layers == other.layers == 1
#            assert table.dim == other.dim
#            assert table.shape == other.shape                 
#
#            name, datakey, headerkeys, scopekeys = table.name, table.datakeys[0], table.headerkeys, table.scopekeys
#            othername, otherdatakey, otherheaderkeys, otherscopekeys = other.name, other.datakeys[0], other.headerkeys, other.scopekeys
#
#            dataarray, variable =  table.dataarrays[datakey], table.variables[datakey]
#            otherdataarray, othervariable = other.dataarray[datakey], other.variables[datakey]
#           
#            newname = kwargs.get('name', namefunction(name, othername))
#            newdatakey = datakey_function(datakey, otherdatakey)
#            newdataarray = dataarray_function(dataarray, otherdataarray, *args, **kwargs)
#            newvariable = variable_function(variable, othervariable, args, kwargs)
#            newscope = update_scope(scope, other, *args, **kwargs)
#
#            for scopekey, scopevalue in newscope.items():
#               if scopekey in table.datakeys: assert scopevalue == _ALLCHAR
#               if scopekey in other.datakeys: assert scopevalue == _ALLCHAR
#               if scopekey in table.headerkeys: assert scopevalue == _ALLCHAR or scopevalue == var.summation(table.headervalues(scopekey, tovarray=True), *args, **kwargs)
#               if scopekey in other.headerkeys: assert scopevalue == _ALLCHAR or scopevalue == var.summation(other.headervalues(scopekey, tovarray=True), *args, **kwargs) 
#
#            newdataset = newdataarray.to_dataset(name=newdatakey) 
#            newheaders = newheaders_function(table, other, *args, newheaderkeys=newheaderkeys, **kwargs)
#            
#            newdataset.attrs = newscope  
#            newvariables = newvariables_function(table, other, *args, newdatakeys=newdatakey, newheaderkeys=newheaderkeys, newscopekeys=newscopekeys, newvariables={newdatakey:newvariable}, **kwargs)            
#            
#            return TableClass(data=, variables=, name=newname)
#
#        update_wrapper(wrapper, function)
#        return wrapper
#    return decorator


#def dataset_operation()
#    def decorator(function): 
#        @operation_loop
#        def wrapper(table, other, *args, **kwargs):
#            TableClass = table.__class__
#            assert table.dim == other.dim
#            assert table.shape == other.shape                 
#
#            name, datakeys, headerkeys, scopekeys = table.name, table.datakeys, table.headerkeys, table.scopekeys
#            othername, otherdatakeys, otherheaderkeys, otherscopekeys = other.name, other.datakeys, other.headerkeys, other.scopekeys
#            
#            newname = kwargs.get('name', name)
#            
#            return TableClass(data=, variables=, name=newname)
#
#        update_wrapper(wrapper, function)
#        return wrapper
#    return decorator            
            

#def newscope_function(table, other, *args, newscopekeys, **kwargs):
#    assert isinstance(newscopekeys, tuple)
#    newscope = ODict()
#    for scopekey in newscopekeys:
#        if all([scopekey in scopekeys for scopekeys in (table.scopekeys, other.scopekeys)]):
#            assert table.scope[scopekey] == other.scope[scopekey]
#            newscope[scopekey] = table.scope[scopekey]
#        elif scopekey in table.scopekeys:
#            newscope[scopekey] = table.scope[scopekey]
#        elif scopekey in other.scopekeys:
#            newscope[scopekey] = other.scope[scopekey]
#        else: raise ValueError(scopekey)
#    for scopekey, scopevalue in newscope.items():
#        if scopekey in table.datakeys: assert scopevalue == _ALLCHAR
#        if scopekey in other.datakeys: assert scopevalue == _ALLCHAR
#        if scopekey in table.headerkeys: assert scopevalue == _ALLCHAR or scopevalue == var.summation(table.headervalues(scopekey, tovarray=True), *args, **kwargs)
#        if scopekey in other.headerkeys: assert scopevalue == _ALLCHAR or scopevalue == var.summation(other.headervalues(scopekey, tovarray=True), *args, **kwargs) 
#    return newscope


#def newvariables_function(table, other, *args, newdatakeys, newheaderkeys, newscopekeys, newvariables, **kwargs):
#    assert isinstance(newvariables, dict)    
#    newdatakeys, newheaderkeys, newscopekeys = [_aslist(item) for item in (newdatakeys, newheaderkeys, newscopekeys)]   
#
#    variables = {}
#    for datakey in newdatakeys: 
#        if datakey in newvariables.keys(): variables[datakey] = newvariables[datakey]
#        else: variables[datakey] = table.variables[datakey] if datakey in table.variables else other.variables[datakey]        
#    
#    assert {table.variables[headerkey] for headerkey in newheaderkeys} == {other.variables[headerkey] for headerkey in newheaderkeys}
#    for headerkey in newheaderkeys: variables[headerkey] = table.variables[headerkey]
#    
#    for scopekey in newscopekeys:
#        if all([scopekey in scopekeys for scopekeys in (table.scopekeys, other.scopekeys)]):
#            assert table.variables[scopekey] == other.variables[scopekey]
#            variables[scopekey] = getvarfunction(scopekey)
#        elif scopekey in table.scopekeys:
#            variables[scopekey] = table.variables[scopekey]
#        elif scopekey in other.scopekeys:
#            variables[scopekey] = other.variables[scopekey]
#        else: raise ValueError(scopekey)
#    return variables


#def layer_function(table, other, *args, name, **kwargs):          
#    TableClass = table.__class__
#    assert table.dim == other.dim
#    assert table.shape == other.shape
#    assert all([datakey not in other.datakeys for datakey in table.datakeys]) 
#
#    name, datakeys, headerkeys, scopekeys = table.name, table.datakeys, table.headerkeys, table.scopekeys
#    othername, otherdatakeys, otherheaderkeys, otherscopekeys = other.name, other.datakeys, other.headerkeys, other.scopekeys
#    
#    dataset, scope, variables = table.dataset.copy(), table.scope.copy(), table.variables.copy()
#    otherdataset, otherscope, othervariables = other.dataset.copy(), other.scope.copy(), other.variables.copy()
#           
#    newdatakeys = (*table.datakeys, *other.datakeys)    
#    newheaderkeys = table.headerkeys
#    newscopekeys = newscopekeys_function(newdatakeys, newheaderkeys, scopekeys, otherscopekeys)
#    
#    variablefunction = lambda keys: {key:(variables[key] if key in variables.keys() else othervariables[key]) for key in keys}   
#    newvariables = variablefunction(newscopekeys)
#    newvariables.update(variablefunction(newheaderkeys))
#    newvariables.update({datakey:variables[datakey] for datakey in table.datakeys})
#    newvariables.update({datakey:othervariables[datakey] for datakey in other.datakeys})
#    
#    newscope = {scopekey:(scope[scopekey] if scopekey in scope else otherscope[scopekey]) for scopekey in newscopekeys}           
#
#    newdataset = xr.merge([dataset, otherdataset])  
#    newdataset.attrs = newscope
#    return TableClass(data=newdataset, variables=newvariables, name=name)  


#@operation(newname_function = lambda name, other: '*'.join([name, other]),
#           newdatakey_function = lambda datakey, other: '*'.join([datakey, other]),
#           newvariable_function = lambda variable, other, args, kwargs: variable.operation(other, *args, method='multiply', **kwargs))
#def multiply(dataarray, otherdataarray, *args, **kwargs):     
#    return dataarray * otherdataarray


#@operation(newname_function = lambda name, other: '/'.join([name, other]),
#           newdatakey_function = lambda datakey, other: '/'.join([datakey, other]),
#           newvariable_function = lambda variable, other, args, kwargs: variable.operation(other, *args, method='multiply', **kwargs))
#def divide(dataarray, otherdataarray, *args, **kwargs):     
#    return dataarray / otherdataarray


#def layer(table, others, *args, name, **kwargs):
#    newtable = table.copy()
#    for other in others: newtable = layer_function(newtable, other, *args, name=name, **kwargs)
#    return newtable
















