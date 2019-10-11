# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 2019
@name    Table Processors Objects
@author: Jack Kriby Cook

"""

from abc import ABC, abstractmethod
import json
from collections import namedtuple as ntuple

from utilities.tree import Node
from utilities.dispatchers import clstype_singledispatcher as typedispatcher

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['FeedPipeline', 'TransformPipeline', 'Display', 'CalculationNode', 'Calculation']
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""


_aslist = lambda items: [items] if not isinstance(items, (list, tuple)) else list(items)   


class Pipeline(ABC):
    def __init__(self, function, tables, parms):
        assert isinstance(parms, dict)
        self.__function = function
        self.__parms = parms
        self.__tables = tables
    
    def __getitem__(self, tablekey):
        def wrapper(*args, **kwargs): return self(tablekey, *args, **kwargs)
        return wrapper 
    
    def __call__(self, tablekey, *args, **kwargs): return self.execute(tablekey, *args, **kwargs)    
    @abstractmethod
    def execute(self, tablekey, *args, **kwargs): pass    
    
    @property
    def parms(self): return self.__parms
    @property
    def tables(self): return self.__tables

    @classmethod
    def create(cls, **kwargs):   
        def decorator(function):  
            return cls(function, **kwargs)
        return decorator          
        

class FeedPipeline(Pipeline):
    def __init__(self, function, tables, parms): 
        assert isinstance(tables, list)
        super().__init__(function, tables, parms)   
        
    def execute(self, tablekey, *args, **kwargs): 
        table = self.__function(*args, **self.__parms, name=tablekey, **kwargs)
        return table

     
class TransformPipeline(Pipeline):        
    def __init__(self, function, tables, parms):   
        assert all([isinstance(item, dict) for item in (tables, parms)])
        assert set(tables.keys()) == set(parms.keys())
        self.__displays = dict()
        super().__init__(function, tables, parms)
        
    def execute(self, tablekey, *args, **kwargs):
        table = self.__function(*args, **self.__parms[tablekey], name=tablekey, **kwargs)
        table.setdisplays(**self.__displays)
        return table        

    @property
    def displays(self): return self.__displays

    def register(self, displaykey):
        def decorator(displayinstance):
            self.__displays[displaykey] = displayinstance
            return displayinstance
        return decorator
    
    
class Display(object):
    def __call__(self, tablekey, table, *args, **kwargs):
        self.__function(table, *args, **self.__parms[tablekey], **kwargs)
    
    def __getitem__(self, tablekey):
        def wrapper(*args, **kwargs): self(tablekey, *args, **kwargs)
        return wrapper 


CalculationKeySgmts = ntuple('CalculationKeySgmts', 'calulationID pipelineID')
class CalculationKey(CalculationKeySgmts):
    calculationkeyformat = '({}){}' 
    def __str__(self): return self.calculationkeyformat.format(self.calulationID, self.pipelineID)


class CalculationNode(Node):
    def __init__(self, calculationKey, pipeline):
        assert isinstance(calculationKey, CalculationKey)
        super().__init__(calculationKey)
        self.__pipeline = pipeline
        self.__table = None
        
    @property
    def calculated(self): return self.__table is not None 
    @property
    def table(self): return self.__table
    
    @property
    def calculationKey(self): return self.key

    @property
    def pipeline(self): return self.__pipeline

    @property
    def tables(self): return [str(child.key) for child in self.children]
    @property
    def displays(self): return [str(displaykey) for displaykey in self.pipeline.displays.keys()]

    def __call__(self, *args, **kwargs): 
        if self.calculated: return self.table
        tables = [child(*args, **kwargs) for child in self.children]
        return self.pipeline[self.calculationKey.pipelineID](*tables, *args, **kwargs)
  

class Calculation(object):
    def __repr__(self): return '{}(calculationID={}, calculationName={})'.format(self.__class__.__name__, self.calculationID, self.calculationName)
    def __init__(self, calculationID, calculationName, treerenderer): 
        self.__calculationID = calculationID
        self.__calculationName = calculationName
        self.__treerenderer = treerenderer
        self.__calculationNodes = {}
         
    @property
    def calculationID(self): return self.__calculationID  
    @property
    def calculationName(self): return self.__calculationName
    
    def showtree(self, tablekey): print(str(self.__treerenderer(self.__calculationNodes[tablekey])), '\n') 
    def __str__(self):
        displays = lambda tablekey: ', '.join(self.__calculationNodes[tablekey].displays)
        tables = lambda tablekey: ', '.join(self.__calculationNodes[tablekey].tables)
        content = {str(key):{'tables':tables(key), 'displays':displays(key)} for key in self.__calculationNodes.keys()}
        content = {key:{k:v for k, v in value.items() if v} for key, value in content.items()}
        
        namestr = ' '.join([self.calculationID.upper(), self.calculationName, self.__class__.__name__])
        jsonstr = json.dumps(content, sort_keys=False, indent=3, separators=(',', ' : '))    
        return ' '.join([namestr, jsonstr])

    def __call__(self, tablekey, *args, **kwargs): return self.__calculationNodes[tablekey](*args, **kwargs)    
    def __getitem__(self, tablekey):
        def wrapper(*args, **kwargs): return self(tablekey, *args, **kwargs)
        return wrapper

    @typedispatcher
    def register(self, pipeline): raise TypeError(type(pipeline))

    @register.register(FeedPipeline)
    def register(self, pipeline):  
        for parent_pipelineID, children_pipelineIDs in pipeline.tables.items():
            parent_calculationKey = CalculationKey(self.calculationID, parent_pipelineID)
            children_calculationKeys = [CalculationKey(self.calculationID, child_pipelineID) for child_pipelineID in _aslist(children_pipelineIDs)]
            
            parent = CalculationNode(parent_calculationKey, pipeline)            
            children = [self.__calculationNodes[str(child_calculationKey)] for child_calculationKey in children_calculationKeys]
            parent.addchildren(*children)            
            self.__calculationNodes[str(parent_calculationKey)] = parent
        return pipeline
    
    @register.register(TransformPipeline)   
    def register(self, pipeline):      
        for parent_pipelineID, children_pipelineIDs in pipeline.tables.items():
            parent_calculationKey = CalculationKey(self.calculationID, parent_pipelineID)
            children_calculationKeys = [CalculationKey(self.calculationID, child_pipelineID) for child_pipelineID in _aslist(children_pipelineIDs)]
            
            parent = CalculationNode(parent_calculationKey, pipeline)            
            children = [self.__calculationNodes[str(child_calculationKey)] for child_calculationKey in children_calculationKeys]
            parent.addchildren(*children)            
            self.__calculationNodes[str(parent_calculationKey)] = parent
        return pipeline
    
    def extend(self, other):
        def decorator(pipeline):
            for parent_pipelineID, children_pipelineIDs in pipeline.tables.items():
                parent_calculationKey = CalculationKey(self.calculationID, parent_pipelineID)
                children_calculationKeys = [CalculationKey(other.calculationID, child_pipelineID) for child_pipelineID in _aslist(children_pipelineIDs)]
                
                parent = CalculationNode(parent_calculationKey, pipeline)            
                children = [other.__calculationNodes[str(child_calculationKey)] for child_calculationKey in children_calculationKeys]
                parent.addchildren(*children)            
                self.__calculationNodes[str(parent_calculationKey)] = parent
            return pipeline
        return decorator        
    













