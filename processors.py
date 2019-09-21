# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 2019
@name    Table Processors Objects
@author: Jack Kriby Cook

"""

import json

from utilities.tree import Node, TreeRenderer

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['Pipeline', 'Display', 'CalculationNode', 'Calculation']
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""


_aslist = lambda items: [items] if not isinstance(items, (list, tuple)) else list(items)   


class Display(object):
    def __repr__(self): return '{}({})'.format(self.__class__.__name__, self.__function)
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
    def __repr__(self): return '{}({}, {})'.format(self.__class__.__name__, self.__function, {key:repr(display) for key, display in self.__displays.items()})
    def __init__(self, function, inputparms_mappings, inputtable_mappings):
        assert all([isinstance(mappings, dict) for mappings in (inputparms_mappings, inputtable_mappings)])
        assert inputparms_mappings.keys() == inputtable_mappings.keys()
        self.__inputParms = inputparms_mappings
        self.__inputTables = inputtable_mappings
        self.__function = function
        self.__displays = {}
        
    @property
    def mapping(self): return self.__inputTables
    @property
    def parms(self): return self.__inputParms

    def __call__(self, tablekey, *args, tables={}, **kwargs):
        assert isinstance(tables, dict)
        inputTables = [tables[inputTable] for inputTable in self.__inputTables[tablekey]]
        inputParms = self.__inputParms[tablekey]
        table = self.__function(*inputTables, *args, **inputParms, name=tablekey, **kwargs)
        table.setdisplays(**self.__displays)
        return table
    
    def __getitem__(self, tablekey):
        def wrapper(*args, tables={}, **kwargs): return self(tablekey, *args, tables=tables, **kwargs)
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


class CalculationNode(Node):
    nameformat = 'Pipeline: {name}'
    tablesformat = 'Tables: ({fromtables}) ==> ({totable})'
    parmsformat = 'Parms: ({})'
    
    def __init__(self, key, pipeline):
        self.__pipeline = pipeline
        super().__init__(key)
    
    @property
    def pipeline(self): return self.__pipeline
    @property
    def parms(self): return self.pipeline.parms[self.key]
    @property
    def tables(self): return self.pipeline.mapping[self.key]    
    
    @property
    def namestr(self): return self.nameformat.format(name=super().__str__())
    @property
    def tablestr(self): return self.tablesformat.format(fromtables=', '.join(self.tables), totable=self.key)
    @property
    def parmstr(self): return self.parmsformat.format(', '.join(['='.join([key, str(value)]) for key, value in self.parms.items()]))
    def __str__(self):  return '\n'.join([self.namestr, self.tablestr, self.parmstr])
    
    def __call__(self, tables, *args, **kwargs): 
        assert isinstance(tables, dict)
        return {self.key:self.pipeline[self.key](*args, tables=tables, **kwargs)}
        

class Calculation(object):
    def __init__(self, name=None, **renderstyle): 
        self.__nodes = {}
        self.__treerenderer = TreeRenderer(**renderstyle)
        self.__name = name
    
    def showtree(self, tablekey): print(str(self.__treerenderer(self.__nodes[tablekey])), '\n')
    
    def __str__(self):
        name = self.__class__.__name__ if not self.__name else '_'.join([self.__name, self.__class__.__name__])
        jsonstr = json.dumps({key:[value.namestr, value.tablestr, value.parmstr] for key, value in self.__nodes.items()}, sort_keys=True, indent=3, separators=(',', ' : '))        
        return ' '.join([name, jsonstr])  
 
    def __call__(self, tablekey, *args, **kwargs):
        tables = {}
        for node in reversed(self.__nodes[tablekey]): 
            if node.key not in tables.keys(): 
                tables.update(node(tables, *args, **kwargs))
        return tables[tablekey]
    
    def __getitem__(self, tablekey):
        def wrapper(*args, **kwargs): return self(tablekey, *args, **kwargs)
        return wrapper

    def register(self, pipeline):
        for parentkey, childrenkeys in pipeline.mapping.items():
            assert isinstance(childrenkeys, list)
            parent = self.__nodes.get(parentkey, CalculationNode(parentkey, pipeline))
            children = [self.__nodes.get(childkey, CalculationNode(childkey, pipeline)) for childkey in childrenkeys]
            parent.addchildren(*children)
            self.__nodes.update({parentkey:parent})
            self.__nodes.update({childkey:child for childkey, child in zip(childrenkeys, children)})           
        return pipeline








