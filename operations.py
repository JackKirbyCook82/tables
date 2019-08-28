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
__all__ = ['add', 'subtract', 'multiply', 'divide', 'average']
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""


_aslist = lambda items: [items] if not isinstance(items, (list, tuple)) else list(items)     


def operation(function):
    @arraytable_operation
    def wrapper(dataarray, other, *args, variables, **kwargs):
        assert isinstance(other, type(dataarray))
        method = function.__name__
        datakey, otherdatakey = dataarray.name, other.name        
        newvariable = variables[datakey].operation(variables[otherdatakey], *args, method=method, **kwargs)
        newdataarray = function(dataarray, other, *args, **kwargs)
        newvariables = {newdataarray.name:newvariable}      
        return newdataarray, newvariables
    update_wrapper(wrapper, function)
    return wrapper

@operation
def add(dataarray, other, *args, **kwargs):    
    newdataarray = dataarray + other    
    newdataarray.name = '+'.join([dataarray.name, other.name]) if dataarray.name != other.name else dataarray.name
    return newdataarray


@operation
def subtract(dataarray, other, *args, **kwargs):
    newdataarray = dataarray - other    
    newdataarray.name = '-'.join([dataarray.name, other.name]) if dataarray.name != other.name else dataarray.name
    return newdataarray


@operation
def multiply(dataarray, other, *args, **kwargs):
    newdataarray = dataarray * other    
    newdataarray.name = '*'.join([dataarray.name, other.name])
    return newdataarray


@operation
def divide(dataarray, other, *args, **kwargs):
    newdataarray = dataarray / other  
    newdataarray.name = '/'.join([dataarray.name, other.name])
    return newdataarray


@operation
def average(dataarray, other, *args, **kwargs):
    newdataarray = dataarray + other
    newdataarray = newdataarray / 2
    if dataarray.name == other.name: newdataarray.name = dataarray.name
    else: newdataarray.name = 'Avg{}&{}'.format(dataarray.name, other.name)
    return newdataarray
            
            

          
            
            
            
            
            









