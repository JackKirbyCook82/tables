# -*- coding: utf-8 -*-
"""
Created on Tues Aug 27 2019
@name    Dataarray Combination Functions
@author: Jack Kriby Cook

"""

from functools import update_wrapper
import xarray as xr

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['merge', 'combine', 'append']
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""


_aslist = lambda items: [items] if not isinstance(items, (list, tuple)) else list(items)     


def combinationloop(function):
    def wrapper(xarray, others, *args, **kwargs):
        newxarray = xarray
        for other in _aslist(others): newxarray = function(xarray, other, *args, **kwargs)
        return newxarray
    update_wrapper(wrapper, function)
    return wrapper


def combination(function):
    @combinationloop
    def wrapper(dataarray, other, *args, axis, **kwargs):
        assert isinstance(other, type(dataarray))
        assert dataarray.name == other.name
        newdataset = function(dataarray, other, *args, axis=axis, **kwargs)
        newdataset.name = dataarray.name
        return newdataset
    update_wrapper(wrapper, function)
    return wrapper


@combination
def merge(dataarray, other, *args, axis, **kwargs):
    dataarray, other = dataarray.expand_dims(axis), other.expand_dims(axis)
    newdataarray = xr.concat([dataarray, other], dim=axis, data_vars='all')
    return newdataarray


@combination
def combine(dataarray, other, *args, axis, **kwargs):
    newdataarray = xr.concat([dataarray, other], dim=axis, data_vars='all') 
    return newdataarray


@combination
def append(dataarray, other, *args, axis, **kwargs):
    other = other.expand_dims(axis)
    newdataarray = xr.concat([dataarray, other], dim=axis, data_vars='all')
    return newdataarray


#@combinationloop
#def layer(xarray, other, *args, axis, **kwargs):
#    assert isinstance(xarray, (xr.DataArray, xr.Dataset))
#    assert isinstance(other, (xr.DataArray, xr.Dataset))
#    newdataset = xr.merge([xarray, other], join='outer')   
#    return newdataset

    
    
    





