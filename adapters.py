# -*- coding: utf-8 -*-
"""
Created on Tues Aug 20 2019
@name:   Table Adapters Decorators
@author: Jack Kirby Cook

"""

from functools import update_wrapper, reduce
import xarray as xr

from tables.alignment import align_arraytables, data_variables, axes_variables

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['flattable_transform', 'arraytable_transform', 'arraytable_operation', 'arraytable_combination']
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""


_aslist = lambda items: [items] if not isinstance(items, (tuple, list)) else list(items)  
_getheader = lambda dataarray, axis, variable: [variable.fromstr(item) for item in dataarray.coords[axis].values]


def flattable_transform(function):
    def wrapper(self, table, *args, column, **kwargs):
        TableClass = table.__class__
        VariablesClass = table.variables.__class__
        
        variables = table.variables.copy()       
        newdataframe, newvariables = function(self, table.dataframe.copy(), variables=table.variables.copy(), *args, column=column, **kwargs)
        variables.update(newvariables)        
        return TableClass(data=newdataframe, variables=VariablesClass(newvariables), name=kwargs.get('name', table.name))
    update_wrapper(wrapper, function)
    return wrapper


def arraytable_transform(function):
    def wrapper(self, table, *args, axis, retag={}, **kwargs):
        assert table.layers == 1
        assert isinstance(retag, dict)
        TableClass = table.__class__
        VariablesClass = table.variables.__class__        
        
        datakey = table.datakeys[0]
        dataarray = table.dataarrays[datakey]
        variables = table.variables.copy()
        datavariable = variables.pop(datakey) 
        axisvariable = variables.pop(axis)
        
        newdataarray, newvariables = function(self, dataarray, *args, axis=axis, variables={axis:axisvariable, datakey:datavariable}, **kwargs)            
        dataarrays = {retag.get(newdataarray.name, newdataarray.name):newdataarray} 
        variables.update(newvariables)
        variables[retag.get(newdataarray.name, newdataarray.name)] = variables.pop(newdataarray.name)

        dataset = xr.merge([dataarray.to_dataset(name=name) for name, dataarray in dataarrays.items()])
        return TableClass(data=dataset, variables=VariablesClass(variables), name=kwargs.get('name', table.name))
    update_wrapper(wrapper, function)
    return wrapper


def arraytable_operation(function):
    def wrapper(table, other, *args, axes=[], **kwargs):
        assert isinstance(other, type(other))
        assert table.layers == other.layers == 1
        TableClass = table.__class__
        VariablesClass = table.variables.__class__
        
        datakey, otherdatakey = table.datakeys[0], other.datakeys[0]
        axes = [*_aslist(kwargs.get('axis', [])), *_aslist(kwargs.get('axes', []))]
        noncoreaxes = [*axes, datakey, otherdatakey]
        
        table, other = align_arraytables(table, other, *args, method='outer', noncoreaxes=noncoreaxes, **kwargs)        
        datavariables = data_variables(table, other, *args, **kwargs)
        variables = axes_variables(table, other, *args, **kwargs)        
        for axis in _aslist(axes):
            other = other.squeeze(axis)
            assert reduce(lambda x, y: x.add(y, *args, **kwargs), table.vheader(axis)) == other.vscope(axis)
        dataarray, otherdataarray = table.dataarrays[datakey], other.dataarrays[otherdatakey]
        
        newdataarray, newvariables = function(dataarray, otherdataarray, *args, variables=datavariables, **kwargs)  
        variables.update(newvariables)        
        dataset = newdataarray.to_dataset()
        return TableClass(data=dataset, variables=VariablesClass(variables), name=kwargs.get('name', table.name))
    update_wrapper(wrapper, function)
    return wrapper


def arraytable_combination(function):
    def wrapper(table, other, *args, **kwargs):
        assert isinstance(other, type(other))
        TableClass = table.__class__
        VariablesClass = table.variables.__class__        

        datakey, otherdatakey = table.datakeys[0], other.datakeys[0]     
        noncoreaxes = [datakey, otherdatakey]
                  
        table, other = align_arraytables(table, other, *args, method='outer', noncoreaxes=noncoreaxes, **kwargs)
        ### Remove table.scope from other.datakeys if scope == ALL & other.scope from table.datakey if scope == ALL
        dataset, otherdataset = table.dataset, other.dataset
        datavariables = data_variables(table, other, *args, **kwargs)
        axesvariables = axes_variables(table, other, *args, **kwargs)        
        assert all([key not in axesvariables.keys() for key in datavariables.keys()])

        newdataset = function(dataset, otherdataset, *args, **kwargs)  
        newvariables = datavariables.copy()
        newvariables.update(axesvariables)
        return TableClass(data=newdataset, variables=VariablesClass(newvariables), name=kwargs.get('name', table.name))
    update_wrapper(wrapper, function)
    return wrapper





















