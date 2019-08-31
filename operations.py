# -*- coding: utf-8 -*-
"""
Created on Sun Jun 2 2019
@name    Operation Functions
@author: Jack Kriby Cook

"""

from functools import update_wrapper

from tables.adapters import arraytable_operation

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['add', 'subtract', 'multiply', 'divide']
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""


_aslist = lambda items: [items] if not isinstance(items, (list, tuple)) else list(items)     


def operation(function):
    @arraytable_operation
    def wrapper(dataarray, other, *args, variables, **kwargs):
        assert isinstance(other, type(dataarray))
        newdataarray, newvariable = function(dataarray, other, *args, variables=variables, **kwargs)
        newvariables = {newdataarray.name:newvariable}  
        return newdataarray, newvariables
    update_wrapper(wrapper, function)
    return wrapper

@operation
def add(dataarray, other, *args, variables, **kwargs):    
    newdataarray = dataarray + other    
    newdataarray.name = '+'.join([dataarray.name, other.name]) if dataarray.name != other.name else dataarray.name
    newvariable = variables[dataarray.name].operation(variables[other.name], *args, method='add', **kwargs)
    return newdataarray, newvariable


@operation
def subtract(dataarray, other, *args, variables, **kwargs):
    newdataarray = dataarray - other    
    newdataarray.name = '-'.join([dataarray.name, other.name]) if dataarray.name != other.name else dataarray.name
    newvariable = variables[dataarray.name].operation(variables[other.name], *args, method='subtract', **kwargs)
    return newdataarray, newvariable


@operation
def multiply(dataarray, other, *args, variables, **kwargs):
    newdataarray = dataarray * other    
    newdataarray.name = '*'.join([dataarray.name, other.name])
    newvariable = variables[dataarray.name].operation(variables[other.name], *args, method='multiply', **kwargs)
    return newdataarray, newvariable


@operation
def divide(dataarray, other, *args, variables, **kwargs):
    newdataarray = dataarray / other  
    newdataarray.name = '/'.join([dataarray.name, other.name])
    newvariable = variables[dataarray.name].operation(variables[other.name], *args, method='divide', **kwargs)
    return newdataarray, newvariable



            
            

          
            
            
            
            
            









