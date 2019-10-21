# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 2019
@name    Table Processors Objects
@author: Jack Kriby Cook

"""

from utilities.tree import Node, Tree, Renderer
from utilities.dispatchers import clstype_singledispatcher as typedispatcher

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['Pipeline', 'Calculation', 'Renderer']
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""


_aslist = lambda items: [items] if not isinstance(items, (list, tuple)) else list(items)   


class Pipeline(Node):
    def __init__(self, key, function, parms, parent=None, children=[]):
        assert isinstance(parms, dict)
        self.__function = function
        self.__parms = parms
        self.__table = None
        super().__init__(key, parent=parent, children=children)

    @property
    def calculated(self): return self.__table is not None 
    @property
    def table(self): return self.__table

    def __call__(self, *args, **kwargs):
        if self.calculated: return self.__table
        tables = [child(*args, **kwargs) for child in self.children]
        print("Running Pipeline: '{}'\n".format(str(self.key)))
        self.__table = self.__function(self.key, *tables, *args, **self.__parms, **kwargs)
        return self.__table


class Calculation(Tree):
    @typedispatcher
    def create(self, nodekeys, *args, parms={}, **kwargs): raise TypeError(type(nodekeys))
    
    @create.register(list, tuple, str)
    def create_feeds(self, nodekeys, *args, parms={}, **kwargs):
        def decorator(function):
            assert isinstance(parms, dict)
            pipelines = [Pipeline(nodekey, function, parms.get(nodekey, {})) for nodekey in nodekeys]
            self.append(*pipelines)
            return function
        return decorator
        
    @create.register(dict)
    def create_transforms(self, nodekeys, *args, parms={}, **kwargs):
        def decorator(function):
            assert isinstance(parms, dict)
            pipelines = [Pipeline(nodekey, function, parms.get(nodekey, {}), children=[self.nodes.get(childkey) for childkey in childrenkeys]) for nodekey, childrenkeys in nodekeys.items()]            
            self.append(*pipelines)
            return function
        return decorator
        

    
    
    
    
    
    
    
    
    
    
    
    
    

    