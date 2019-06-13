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
__all__ = ['add', 'subtract', 'multiply', 'divide', 'combine', 'merge', 'append', 'OPERATIONS']
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""


OPERATIONS = {}

def operation(function):
    def wrapper(table, other, *args, **kwargs):
        TableClass = table.__class__
        xarray = function(table.xarray, other.xarray, *args, **kwargs)        
        return TableClass(xarray, name=table.name)
    update_wrapper(wrapper, function)
    OPERATIONS[function.__name__] = wrapper
    return wrapper
        

class UnderConstructionError(Exception): pass


@operation
def add(xarray, other, *args, **kwargs):
    raise UnderConstructionError('tables.transformation.add(table, other, *args, **kwargs)')

@operation
def subtract(xarray, other, *args, **kwargs):
    raise UnderConstructionError('tables.transformation.subtract(table, other, *args, **kwargs)')

@operation
def multiply(xarray, other, *args, **kwargs):
    raise UnderConstructionError('tables.transformation.multiply(table, other, *args, **kwargs)')

@operation
def divide(xarray, other, *args, **kwargs):
    raise UnderConstructionError('tables.transformation.divide(table, other, *args, **kwargs)')

@operation
def combine(xarray, other, *args, onscope, **kwargs):
    return xr.concat([xarray, other], pd.Index([xarray.attrs[onscope], other.attrs[onscope]], name=onscope))

@operation
def merge(xarray, other, *args, onaxis, **kwargs):
    return xr.concat([xarray, other], dim=onaxis)

@operation
def append(xarray, other, *args, toaxis, **kwargs):
    other = other.expand_dims(toaxis)
    other.coords[toaxis] = pd.Index([other.attrs.pop(toaxis)], name=toaxis)
    return xr.concat([xarray, other], dim=toaxis)







