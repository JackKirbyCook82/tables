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
__all__ = []
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""


OPERATIONS = {}

def operation(mainfunction):    
    _registry = {}
    
    def register(key): 
        def register_decorator(regfunction): 
            _registry[key] = regfunction
            def register_wrapper(*args, **kwargs): 
                return regfunction(*args, **kwargs) 
            return register_wrapper 
        return register_decorator 
    
    def wrapper(table, other, *args, **kwargs):
        TableClass = table.__class__
        operatefunc = lambda func, key: func(getattr(table, key), getattr(other, key), *args, dim=table.dim, shape=table.shape, **kwargs)
        operated = {key:operatefunc(function, key) for key, function in _registry.items()}
        contents = table.todict()
        contents.update(operated)
        contents.update({'name':kwargs.get('name', table.name)})
        return TableClass(**contents)
    
    wrapper.register = register 
    update_wrapper(wrapper, mainfunction)
    OPERATIONS[mainfunction.__name__] = wrapper
    return wrapper


@operation
def multiply(*args, **kwargs): pass

@multiply.register('data')
def multiply_data(xarray, other, *args, **kwargs): 
    assert all([xarray.attrs[key] == other.attrs[key] for key in xarray.attrs.keys() if key in other.attrs.keys()])   
    newxarray = xarray * other
    scope = xarray.attrs
    scope.update(other.attrs)
    newxarray.attrs = scope
    return newxarray

@multiply.register('datakey')
def multiply_datakey(datakey, other, *args, **kwargs):
    return '*'.join([datakey, other])

@multiply.register('variables')
def multiply_variables(variables, others, *args, dim, **kwargs):
    VariablesClass = variables.__class__
    datakey = '*'.join([list(variables.keys())[0], list(others.keys())[0]])
    datavar = variables[0].operation(variables[0], *args, method='multiply', **kwargs)
    hdrvars = variables[slice(1, 1+dim)]
    assert hdrvars == others[slice(1, 1+dim)]
    scopevars = variables[slice(1+dim, None)].update(others[slice(1+dim, None)])
    return VariablesClass([(key, value) for key, value in zip(chain((datakey,), hdrvars.keys(), scopevars.keys()), chain((datavar,), hdrvars.values(), scopevars.values()))])


@operation
def divide(*args, **kwargs): pass

@divide.register('data')
def divide_data(xarray, other, *args, **kwargs):
    assert all([xarray.attrs[key] == other.attrs[key] for key in xarray.attrs.keys() if key in other.attrs.keys()])    
    newxarray = xarray / other 
    scope = xarray.attrs
    scope.update(other.attrs)
    newxarray.attrs = scope
    return newxarray

@divide.register('datakey')
def divide_datakey(datakey, other, *args, **kwargs):
    return '/'.join([datakey, other])

@divide.register('variables')
def divide_variables(variables, others, *args, dim, **kwargs):
    VariablesClass = variables.__class__
    datakey = '/'.join([list(variables.keys())[0], list(others.keys())[0]])
    datavar = variables[0].operation(variables[0], *args, method='divide', **kwargs)
    hdrvars = variables[slice(1, 1+dim)]
    assert hdrvars == others[slice(1, 1+dim)]
    scopevars = variables[slice(1+dim, None)].update(others[slice(1+dim, None)])
    return VariablesClass([(key, value) for key, value in zip(chain((datakey,), hdrvars.keys(), scopevars.keys()), chain((datavar,), hdrvars.values(), scopevars.values()))])

@operation
def combine(*args, **kwargs): pass

@combine.register('data')
def combine_data(xarray, other, *args, onscope, **kwargs):
    newxarray = xr.concat([xarray, other], pd.Index([xarray.attrs[onscope], other.attrs[onscope]], name=onscope))
    return {'data':newxarray}


@operation
def merge(*args, **kwargs): pass

@merge.register('data')
def merge_data(xarray, other, *args, onaxis, **kwargs):
    newxarray = xr.concat([xarray, other], dim=onaxis)
    return {'data':newxarray}


@operation
def append(*args, **kwargs): pass

@append.register('data')
def append_data(xarray, other, *args, toaxis, **kwargs):
    other = other.expand_dims(toaxis)
    other.coords[toaxis] = pd.Index([other.attrs.pop(toaxis)], name=toaxis)
    newxarray = xr.concat([xarray, other], dim=toaxis)
    return {'data':newxarray}







