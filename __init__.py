# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 2019
@name:   Table Objects
@author: Jack Kirby Cook

"""

from abc import ABC

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['ArrayTable', 'FlatTable']
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""



class TableBase(ABC):
    pass


class ArrayTable(TableBase):
    def __init__(self, xarray, specs={}):
        pass
    
    def __str__(self):
        pass
    
    def flatten(self): 
        pass
    
    
class FlatTable(TableBase):
    def __init__(self, dataframe, specs={}):
        pass

    def __str__(self):
        pass

    def unflatten(self, datakey):
        pass