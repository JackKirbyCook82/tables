# -*- coding: utf-8 -*-
"""
Created on Tues Aug 27 2019
@name    Combination Functions
@author: Jack Kriby Cook

"""

from functools import update_wrapper
import xarray as xr

from tables.adapters import arraytable_combination

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['layer']
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""


_aslist = lambda items: [items] if not isinstance(items, (list, tuple)) else list(items)     


def combination(function):
    @arraytable_combination
    def wrapper(dataset, other, *args, **kwargs):
        assert isinstance(other, type(dataset))
        newdataset = function(dataset, other, *args, **kwargs)
        return newdataset
    update_wrapper(wrapper, function)
    return wrapper


@combination
def layer(dataset, other, *args, **kwargs):
    newdataset = xr.merge([dataset, other])
    return newdataset
    