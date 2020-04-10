# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 2019
@name    Table Processors Objects
@author: Jack Kriby Cook

"""

import json
from copy import copy

from utilities.tree import Node, Tree, Renderer

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['Pipeline', 'CalculationProcess', 'CalculationRenderer']
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
    @property
    def parms(self): return self.__parms

    def __str__(self): 
        content = {'tables':[str(child.key) for child in self.children], 'parms':self.__parms}
        return json.dumps(content, sort_keys=False, indent=3, separators=(',', ' : '), default=str)  

    def __call__(self, *args, **kwargs):
        if self.calculated: return self.__table
        tables = [child(*args, **kwargs) for child in self.children]
        print("Running Pipeline: '{}'\n".format(str(self.key)))
        self.__table = self.__function(self.key, *tables, *args, **self.__parms, **kwargs)
        self.__table.rename(self.__parms.get('name', self.key))
        return self.__table


class Calculation(Tree):  
    def __str__(self):
        namestr = '{} ("{}")'.format(self.name if self.name else self.__class__.__name__, self.key)
        content = {key:[str(child.key) for child in pipeline.children] for key, pipeline in iter(self)}        
        jsonstr = json.dumps(content, sort_keys=False, indent=3, separators=(',', ' : '), default=str)  
        return ' '.join([namestr, jsonstr])  


class CalculationRenderer(Renderer): pass


class CalculationProcess(dict):
    def __init__(self, key, *args, name=None, **kwargs):
        self.__key, self.__name = key, name
        self.__pipelines = []
        super().__init__()
    
    def __repr__(self): 
        if self.name: return "{}(key='{}', name='{}')".format(self.__class__.__name__, self.key, self.name)
        else: return "{}(key='{}')".format(self.__class__.__name__, self.key)  
    
    @property
    def name(self): return self.__name
    @property
    def key(self): return self.__key
    
    def create(self, **kwargs):
        def decorator(function):
            pipelines = [Pipeline(key, function, value.get('parms', {})) for key, value in kwargs.items()]
            queue = {key:_aslist(value.get('tables', [])) for key, value in kwargs.items()}
            assert not any([key in self.keys() for key in queue.keys()])
            self.__pipelines = [*self.__pipelines, *pipelines]
            self.update(queue)
            return function
        return decorator    

    def retag(self, key=None, name=None): 
        self.__key =  key if key else self.key
        self.__name = name if name else self.name 
        
    def copy(self, key, *args, name=None, **kwargs): 
        instance = copy(self)
        instance.retag(key=key, name=name)
        return instance

    def __call__(self, *args, **kwargs):
        nodes = {pipeline.key:pipeline for pipeline in self.__pipelines}
        for nodekey, childrenkeys in self.items(): 
            nodes[nodekey].addchildren(*[nodes[childkey] for childkey in childrenkeys if nodes[childkey] not in nodes[nodekey].children])        
        return Calculation(self.key, name=self.name, nodes=nodes)
















