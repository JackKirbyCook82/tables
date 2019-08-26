# -*- coding: utf-8 -*-
"""
Created on Tues Aug 20 2019
@name:   Table Adapters Decorators
@author: Jack Kirby Cook

"""

from functools import update_wrapper, reduce
import xarray as xr

from tables.alignment import align_arraytables, align_variables

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['flattable_transform', 'arraytable_transform', 'arraytable_operation']
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""


_aslist = lambda items: [items] if not isinstance(items, (tuple, list)) else list(items)  
_getheader = lambda dataarray, axis, variable: [variable.fromstr(item) for item in dataarray.coords[axis].values]


def flattable_transform(function):
    def wrapper(self, table, *args, column, **kwargs):
        TableClass = table.__class__
        VariableClass = table.variables.__class__
        
        variables = table.variables.copy()       
        newdataframe, newvariables = function(self, table.dataframe.copy(), variables=table.variables.copy(), *args, column=column, **kwargs)
        variables.update(newvariables)        
        return TableClass(data=newdataframe, variables=VariableClass(newvariables), name=kwargs.get('name', table.name))
    update_wrapper(wrapper, function)
    return wrapper


def arraytable_transform(function):
    def wrapper(self, table, *args, axis, **kwargs):
        assert table.layers == 1
        TableClass = table.__class__
        VariableClass = table.variables.__class__        
        
        datakey = table.datakeys[0]
        dataarray = table.dataarrays[datakey]
        variables = table.variables.copy()
        datavariable = variables.pop(datakey) 
        axisvariable = variables.pop(axis)
        
        newdataarray, newvariables = function(self, dataarray, *args, axis=axis, variables={axis:axisvariable, datakey:datavariable}, **kwargs)            
        dataarrays = {newdataarray.name:newdataarray} 
        variables.update(newvariables)

        dataset = xr.merge([dataarray.to_dataset(name=name) for name, dataarray in dataarrays.items()])
        return TableClass(data=dataset, variables=VariableClass(variables), name=kwargs.get('name', table.name))
    update_wrapper(wrapper, function)
    return wrapper


def arraytable_operation(function):
    def wrapper(table, other, *args, axis, **kwargs):
        assert isinstance(other, type(other))
        assert table.layers == other.layers == 1
        TableClass = table.__class__
        VariableClass = table.variables.__class__

        table, other = align_arraytables(table, other, *args, method='outer', noncoreaxes=axis, **kwargs)
        variables = align_variables(table.variables, other.variables)

        datakey, otherdatakey = table.datakeys[0], other.datakeys[0]
        if axis:
            other = other.squeeze(axis)
            assert reduce(lambda x, y: x.add(y, *args, **kwargs), table.vheader(axis)) == other.vscope(axis)
        dataarray, otherdataarray = table.dataarrays[datakey], other.dataarrays[otherdatakey]
        
        newdataarray, newvariables = function(dataarray, otherdataarray, *args, variables=variables, **kwargs)  
        variables.update(newvariables)        
        dataset = newdataarray.to_dataset()
        return TableClass(data=dataset, variables=VariableClass(variables), name=kwargs.get('name', table.name))
    update_wrapper(wrapper, function)
    return wrapper















