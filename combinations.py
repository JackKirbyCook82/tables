# -*- coding: utf-8 -*-
"""
Created on Tues Aug 27 2019
@name    Dataarray Combination Functions
@author: Jack Kriby Cook

"""

from functools import update_wrapper
import xarray as xr

from tables.adapters import arraytable_combine, arraytable_layer

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['merge', 'concat', 'append']
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""


_RECONCILIATION = {'avg': lambda x, y: (x + y)/2, 'max': lambda x, y: max([x, y]), 'min': lambda x, y: min([x, y])}

_aslist = lambda items: [items] if not isinstance(items, (list, tuple)) else list(items)     


def combinationloop(function):
    def wrapper(xarray, others, *args, **kwargs):
        newxarray = xarray
        for other in _aslist(others): newxarray = function(newxarray, other, *args, **kwargs)
        return newxarray
    update_wrapper(wrapper, function)
    return wrapper


def combination(function):
    @combinationloop
    def wrapper(xarray, other, *args, axis=None, axes=[], **kwargs):
        assert isinstance(xarray, type(other))
        axes = [item for item in [*_aslist(axis), *_aslist(axes)] if item is not None]
        if len(axes) == 0: newxarray = function(xarray, other, *args, **kwargs)
        elif len(axes) == 1: newxarray = function(xarray, other, *args, axis=axes[0], **kwargs)
        else: newxarray = function(xarray, other, *args, axes=axes, **kwargs)
        return newxarray
    update_wrapper(wrapper, function)
    return wrapper


@arraytable_combine
@combination
def merge(dataarray, other, *args, axis, **kwargs):
    dataarray, other = dataarray.expand_dims(axis), other.expand_dims(axis)
    newdataarray = xr.concat([dataarray, other], dim=axis, data_vars='all')
    return newdataarray


@arraytable_combine
@combination
def concat(dataarray, other, *args, axis, **kwargs):
    newdataarray = xr.concat([dataarray, other], dim=axis, data_vars='all') 
    return newdataarray


@arraytable_combine
@combination
def append(dataarray, other, *args, axis, **kwargs):  
    other = other.expand_dims(axis)
    newdataarray = xr.concat([dataarray, other], dim=axis, data_vars='all')
    return newdataarray


@arraytable_layer
@combination
def layer(xarray, other, *args, **kwargs):
    newdataset = xr.merge([xarray, other], join='outer')   
    return newdataset



    





