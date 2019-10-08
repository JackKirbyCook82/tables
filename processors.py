# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 2019
@name    Table Processors Objects
@author: Jack Kriby Cook

"""

import json
from collections import namedtuple as ntuple

from utilities.tree import Node

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['Pipeline', 'Display', 'CalculationNode', 'Calculation']
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""


_aslist = lambda items: [items] if not isinstance(items, (list, tuple)) else list(items)   


class Display(object):
    def __init__(self, function, inputparms_mapping): 
        assert isinstance(inputparms_mapping, dict)
        self.__inputParms = inputparms_mapping
        self.__function = function
      
    @property
    def parms(self): return self.__inputParms
    
    def __call__(self, tablekey, table, *args, **kwargs):
        inputParms = self.__inputParms[tablekey]
        self.__function(table, *args, **inputParms, **kwargs)
    
    def __getitem__(self, tablekey):
        def wrapper(*args, **kwargs): return self(tablekey, *args, **kwargs)
        return wrapper 

    @classmethod
    def create(cls, **mapping):
        def decorator(function):
            inputparms_mapping = mapping
            return cls(function, inputparms_mapping)
        return decorator


class Pipeline(object):        
    def __init__(self, function, inputparms_mappings, inputtable_mappings):
        assert all([isinstance(mappings, dict) for mappings in (inputparms_mappings, inputtable_mappings)])
        assert inputparms_mappings.keys() == inputtable_mappings.keys()
        self.__inputParms = inputparms_mappings
        self.__inputTables = inputtable_mappings
        self.__function = function
        self.__displays = dict()
        
    @property
    def tables(self): return self.__inputTables
    @property
    def parms(self): return self.__inputParms
    @property
    def displays(self): return self.__displays
    
    def __iter__(self): 
        for outputTable, inputTables in self.__inputTables.items(): yield outputTable, inputTables

    def __call__(self, tablekey, *args, **kwargs):
        inputParms = self.__inputParms[tablekey]
        table = self.__function(*args, **inputParms, name=tablekey, **kwargs)
        table.setdisplays(**self.__displays)
        return table
    
    def __getitem__(self, tablekey):
        def wrapper(*args, **kwargs): return self(tablekey, *args, **kwargs)
        return wrapper 
              
    @classmethod
    def create(cls, **mappings):
        def decorator(function):
            inputparms_mappings = {key:values['parms'] for key, values in mappings.items()}
            inputtable_mappings = {key:_aslist(values.get('tables', [])) for key, values in mappings.items()}   
            return cls(function, inputparms_mappings, inputtable_mappings)
        return decorator 

    def register(self, displaykey):
        def decorator(displayinstance):
            self.__displays[displaykey] = displayinstance
            return displayinstance
        return decorator


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

    def register(self, pipeline):
        assert isinstance(pipeline, Pipeline)     
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
    













