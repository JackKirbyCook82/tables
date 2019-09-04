# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 2019
@name    Processors Objects
@author: Jack Kriby Cook

"""

from abc import ABC, abstractmethod

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['Feed', 'Transform', 'Calculation']
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""


_aslist = lambda items: [items] if not isinstance(items, (list, tuple)) else list(items)   
_filterna = lambda items: [item for item in items if item]


class Pipeline(ABC):
    def __init__(self, function, **kwargs): 
        assert all([isinstance(values, dict) for values in kwargs.values()])
        self.__tables = {key:_filterna(_aslist(values.get('tables', None))) for key, values in kwargs.items()}
        self.__parms = {key:values['parms'] for key, values in kwargs.items()}
        self.__function = function
        
    @property
    def function(self): return self.__function
    @property
    def tables(self): return self.__tables
    @property
    def parms(self): return self.__parms
    
    def __call__(self, tablekey, *args, **kwargs):
        table = self.execute(tablekey, *args, **kwargs)
        return {tablekey:table}
    
    @abstractmethod
    def execute(self, tablekey, *args, **kwargs): pass


class Feed(Pipeline):
    def __init__(self, webapi, function, **kwargs):
        self.webapi = webapi
        super().__init__(function, **kwargs)
 
    def setwebapi(self, *args, universe, index, header, scope, **kwargs):
        assert isinstance(scope, dict)
        self.webapi.reset()
        self.webapi.setuniverse(universe)
        self.webapi.setindex(index)
        self.webapi.setheader(header)
        self.webapi.setitems(**scope)
        
    def execute(self, tablekey, *args, **kwargs): 
        parms = self.parms[tablekey]
        self.setwebapi(**parms)
        dataframe = self.webapi(*args, **kwargs)
        table = self.function(dataframe, *args, **parms, name=tablekey, **kwargs)
        return table


class Transform(Pipeline):
    def execute(self, tablekey, *args, tables, **kwargs):
        parms = self.parms[tablekey]
        requiredtables = [tables[required_tablekey] for required_tablekey in self.tables[tablekey]]        
        table = self.function(*requiredtables, *args, **parms, name=tablekey, **kwargs)
        return table


class Calculation(dict):
    def addfeed(self, mapping, webapi):
        def decorator(function):
            feed = Feed(webapi, function, **mapping)
            ###
            return feed
        return decorator
    
    def addtransform(self, mapping):
        def decorator(function):
            transform = Transform(function, **mapping)
            ###
            return transform
        return decorator




    
    

















