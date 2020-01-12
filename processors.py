# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 2019
@name    Table Processors Objects
@author: Jack Kriby Cook

"""

import json

from utilities.tree import Node, Tree, Renderer

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


class FrozenCalculationError(Exception): pass

def freeze(function):
    def wrapper(self, *args, **kwargs):
        if self.frozen: raise FrozenCalculationError(repr(self))
        else: return function(self, *args, **kwargs)
    return wrapper
    

class Calculation(Tree):  
    @property
    def frozen(self): return self.__frozen
    @property
    def queue(self): return self.__queue
    
    def __init__(self, *args, **kwargs):
        self.__queue = {}
        self.__frozen = False
        super().__init__(*args, **kwargs)
    
    def __str__(self):
        namestr = '{} ("{}")'.format(self.name if self.name else self.__class__.__name__, self.key)
        if self.frozen: content = {key:[str(child.key) for child in pipeline.children] for key, pipeline in iter(self)}        
        else: content = {key:self.__queue[key] for key, pipeline in iter(self)}  
        jsonstr = json.dumps(content, sort_keys=False, indent=3, separators=(',', ' : '), default=str)  
        return ' '.join([namestr, jsonstr])

    def create(self, **kwargs):
        def decorator(function):
            pipelines = [Pipeline(key, function, value.get('parms', {})) for key, value in kwargs.items()]
            queue = {key:_aslist(value.get('tables', [])) for key, value in kwargs.items()}
            assert not any([key in self.__queue.keys() for key in queue.keys()])
            super(Calculation, self).append(*pipelines)
            self.__queue.update(queue)
            return function
        return decorator
    
    def append(self, *args, **kwargs): raise NotImplementedError('{}.{}()'.format(self.__class__.__name__, 'append'))    
    
    @freeze
    def __iadd__(self, other):
        assert isinstance(other, type(self))   
        assert not any([otherkey in self.queue.keys() for otherkey in other.queue.keys()])
        assert not any([otherkey in [nodekey for nodekey, node in iter(self)] for otherkey, other in iter(other)])                
        self.__queue.update(other.queue)
        super(Calculation, self).append(*[node for nodekey, node in iter(other)])
        return self  

    @freeze
    def __call__(self, *args, **kwargs):
        for nodekey, childrenkeys in self.queue.items(): 
            self[nodekey].addchildren(*[self[childkey] for childkey in childrenkeys if self[childkey] not in self[nodekey].children])
        self.__queue = {}
        self.__frozen = True
        return



        


















