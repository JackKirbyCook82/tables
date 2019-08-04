# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 2019
@name    Calculation Objects
@author: Jack Kriby Cook

"""

import tables as tbls

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['Feed', 'Pipeline', 'Calculation']
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""


class Feed(dict): 
    def __init__(self, webapi, variables):
        self.webapi = webapi
        self.variables = variables

    def __setitem__(self, tablekey, value):
        assert isinstance(value, dict)
        super().__setitem__(tablekey, value)

    def setwebapi(self, universe, index, header, *args, scope={}, **kwargs):
        self.webapi.reset()      
        self.webapi.setitems(universe=universe, index=index, header=header, **scope)  
        print(self.webapi.tables, '\n')     

    def tablekeys(self, universe, index, header, *args, axes=(), scope={}, **kwargs):
        assert isinstance(axes, tuple)
        assert isinstance(scope, dict)
        datakeys = [universe]
        headerkeys = [index, header, *axes]
        scopekeys = list(scope.keys())        
        return [datakeys, headerkeys, scopekeys]
    
    def __call__(self, tablekey, *args, **kwargs):
        items = self[tablekey]
        name = items.pop('name', self.webapi.__class__.__name__)
        self.setwebapi(**items)
        dataframe = self.webapi(*args, **kwargs)
        flattable = tbls.FlatTable(data=dataframe, variables=self.variables, name=name)
        arraytable = flattable.unflatten(*self.tablekeys(**items))
        for axis in arraytable.headerkeys: arraytable = arraytable.sort(axis, ascending=True)
        return arraytable


class Pipeline(dict):
    def __init__(self, feed, function):
        self.feed = feed
        self.function = function
        
    def __setitem__(self, tablekey, value):
        assert isinstance(value, dict)
        super().__setitem__(tablekey, value)
        
    def __call__(self, tablekey, *args, **kwargs):
        items = self[tablekey]
        arraytable = self.feed(tablekey, *args, **kwargs)
        arraytable = self.function(arraytable, *args, **items, **kwargs)
        return arraytable


class Calculation(dict):
    def __init__(self, pipeline, function):
        self.pipeline = pipeline
        self.function = function
        
    def __setitem__(self, calculationkey, value):
        assert isinstance(value, dict)
        super().__setitem__(calculationkey, value)
        
    def __call__(self, calculationkey, *args, **kwargs):
        items = self[calculationkey]
        tablekeys = items.pop('tablekeys')
        tables = {tablekey:table for tablekey, table in self.runpipeline(tablekeys, *args, **kwargs)}
        arraytable = self.function(*[tables[tablekey] for tablekey in tablekeys], *args, **items, **kwargs)
        return arraytable
    
    def runpipeline(self, tablekeys, *args, **kwargs):
        assert isinstance(tablekeys, tuple)
        for tablekey in tablekeys: yield tablekey, self.pipeline(tablekey, *args, **kwargs)
        
    
        




    
    

















