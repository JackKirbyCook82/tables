# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 2019
@name    Processors Objects
@author: Jack Kriby Cook

"""

from functools import update_wrapper
import json

from utilities.tree import Node, TreeRenderer

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['tableprocessor', 'Pipeline', 'Calculation']
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""


_aslist = lambda items: [items] if not isinstance(items, (list, tuple)) else list(items)   


def tableprocessor(**tables):
    def decorator(function):
        parms_mapping = {key:values['parms'] for key, values in tables.items()}
        table_mapping = {key:_aslist(values.get('tables', [])) for key, values in tables.items()}
        
        def wrapper(tablekey, *args, tables={}, **kwargs):
            assert isinstance(tables, dict)
            tables = [tables[input_tablekey] for input_tablekey in table_mapping[tablekey]]
            parms = parms_mapping[tablekey]
            return function(*tables, *args, **parms, name=tablekey, **kwargs)        
        
        wrapper.registry = table_mapping
        wrapper.parms = parms_mapping
        update_wrapper(wrapper, function)
        return wrapper    
    return decorator


class Pipeline(Node):
    nameformat = 'Pipeline: {name}'
    tablesformat = 'Tables: ({fromtables}) ==> ({totable})'
    parmsformat = 'Parms: ({})'
    
    def __init__(self, key, function):
        self.__function = function
        super().__init__(key)
    
    @property
    def function(self): return self.__function
    @property
    def parms(self): return self.function.parms[self.key]
    @property
    def tables(self): return self.function.registry[self.key]    
    
    @property
    def namestr(self): return self.nameformat.format(name=super().__str__())
    @property
    def tablestr(self): return self.tablesformat.format(fromtables=', '.join(self.tables), totable=self.key)
    @property
    def parmstr(self): return self.parmsformat.format(', '.join(['='.join([key, str(value)]) for key, value in self.parms.items()]))
    def __str__(self):  return '\n'.join([self.namestr, self.tablestr, self.parmstr])
    
    def __call__(self, tables, *args, **kwargs): 
        assert isinstance(tables, dict)
        return {self.key:self.function(self.key, *args, tables=tables, **kwargs)}


class Calculation(dict):
    def __init__(self, renderstyle): 
        self.__treerenderer = TreeRenderer(**renderstyle)
    
    def register(self, function):
        for parentkey, childrenkeys in function.registry.items():
            assert isinstance(childrenkeys, list)
            parent = self.get(parentkey, Pipeline(parentkey, function))
            children = [self.get(childkey, Pipeline(childkey, function)) for childkey in childrenkeys]
            parent.addchildren(*children)
            self.update({parentkey:parent})
            self.update({childkey:child for childkey, child in zip(childrenkeys, children)})           
        return function
    
    def __call__(self, tablekey, *args, **kwargs):
        tables = {}
        for node in reversed(self[tablekey]): 
            if node.key not in tables.keys(): 
                tables.update(node(tables, *args, **kwargs))
        return tables[tablekey]
    
    def tree(self, tablekey): return self.__treerenderer(self[tablekey])
    def __str__(self):
        calcstring = json.dumps({key:[value.namestr, value.tablestr, value.parmstr] for key, value in self.items()}, sort_keys=True, indent=3, separators=(',', ' : '))
        return 'Calculations {}'.format(calcstring)

















