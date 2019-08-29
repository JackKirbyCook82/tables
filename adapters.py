# -*- coding: utf-8 -*-
"""
Created on Tues Aug 20 2019
@name:   Table Adapters Decorators
@author: Jack Kirby Cook

"""

from functools import update_wrapper, reduce

from tables.alignment import align_arraytables, data_variables, axes_variables

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['flattable_transform', 'arraytable_inversion', 'arraytable_transform', 'arraytable_operation']
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""


_aslist = lambda items: [items] if not isinstance(items, (tuple, list)) else list(items)  
_getheader = lambda dataarray, axis, variable: [variable.fromstr(item) for item in dataarray.coords[axis].values]


def retag(function):
    def wrapper(*args, retag={}, **kwargs):
        assert isinstance(retag, dict)
        table = function(*args, **kwargs)
        table = table.retag(**retag) if retag else table
        return table
    update_wrapper(wrapper, function)
    return wrapper
        

def flattable_transform(function):
    @retag
    def wrapper(self, table, *args, **kwargs):
        TableClass = table.__class__
        VariablesClass = table.variables.__class__
           
        dataframe, variables = table.dataframe, table.variables.copy()
        newdataframe, newvariables = function(self, dataframe, variables=variables, *args, **kwargs)
        variables.update(newvariables)        
        
        return TableClass(data=newdataframe, variables=VariablesClass(variables), name=kwargs.get('name', table.name))    
    update_wrapper(wrapper, function)
    return wrapper


def arraytable_inversion(function):
    @retag
    def wrapper(self, table, *args, **kwargs):
        assert table.layers == 1
        TableClass = table.__class__
        VariablesClass = table.variables.__class__        
                
        dataarray, variables = list(table.dataarrays.values())[0], table.variables.copy()
        newdataarray, newvariables = function(self, dataarray, *args, variables=variables, **kwargs)
        newdataset = newdataarray.to_dataset()
        variables.update(newvariables)
        
        return TableClass(data=newdataset, variables=VariablesClass(variables), name=kwargs.get('name', table.name))
    update_wrapper(wrapper, function)
    return wrapper


def arraytable_transform(function):
    @retag
    def wrapper(self, table, *args, **kwargs):
        TableClass = table.__class__
        VariablesClass = table.variables.__class__        
        
        dataset, variables = table.dataset, table.variables.copy()
        newdataset, newvariables = function(self, dataset, *args, variables=variables, **kwargs)
        variables.update(newvariables)
        
        return TableClass(data=newdataset, variables=VariablesClass(variables), name=kwargs.get('name', table.name))
    update_wrapper(wrapper, function)
    return wrapper


def arraytable_operation(function):
    def wrapper(table, other, *args, axes=[], **kwargs):
        assert isinstance(other, type(table))
        assert table.layers == other.layers == 1
        TableClass = table.__class__
        VariablesClass = table.variables.__class__    
        
        datakey, otherdatakey = table.datakeys[0], other.datakeys[0]
        axes = [item for item in [*_aslist(kwargs.get('axis', None)), *_aslist(axes)] if item is not None]
        noncoreaxes = [*axes, datakey, otherdatakey]
        
        table, other = align_arraytables(table, other, *args, method='left', noncoreaxes=noncoreaxes, **kwargs)        
        datavariables = data_variables(table, other, *args, **kwargs)
        variables = axes_variables(table, other, *args, **kwargs)        
        for axis in axes:
            other = other.squeeze(axis)
            assert reduce(lambda x, y: x.add(y, *args, **kwargs), table.vheader(axis)) == other.vscope(axis)
        
        dataarray, otherdataarray = table.dataarrays[datakey], other.dataarrays[otherdatakey]
        newdataarray, newvariables = function(dataarray, otherdataarray, *args, variables=datavariables, **kwargs)  
        variables.update(newvariables)        
        newdataset = newdataarray.to_dataset()        
        
        return TableClass(data=newdataset, variables=VariablesClass(variables), name=kwargs.get('name', table.name))
    update_wrapper(wrapper, function)
    return wrapper


#def arraytable_compilation(function):
#    def wrapper(tables, *args, **kwargs):
#        assert set([type(table) for table in _aslist(tables)])
#        assert all([table.layers == 1 for table in _aslist(tables)])
#        tables = _aslist(tables)
#        if len(tables) <= 1: return tables[0]
#        TableClass = tables[0].__class__
#        VariablesClass = tables[0].variables.__class__    
#      
#        tables = [tables[0]] + [align_arraytables(tables[0], other, *args, method='left', **kwargs) for other in tables[1:]] 
#        datavariableitems = [data_variables(tables[0], other, *args, **kwargs) for other in tables[1:]]  
#        assert set(datavariableitems) == 1
#        datavariables = datavariableitems[0]
#        variableitems = [axes_variables(tables[0], other, *args, **kwargs) for other in tables[1:]]  
#        assert set(variableitems) == 1
#        variables = variableitems[0]
#
#        dataarrays = [table.dataarrays[table.datakeys[0]] for table in tables]
#        newdataarray, newvariables = function(dataarrays, *args, variables=datavariables, **kwargs)
#        variables.update(newvariables)
#        newdataset = newdataarray.to_dataset()
#
#        return TableClass(data=newdataset, variables=VariablesClass(variables), name=kwargs.get('name', tables[0].name))
#    update_wrapper(wrapper, function)
#    return wrapper


#def arraytable_combination(function):
#    def wrapper(table, others, *args, **kwargs):
#        assert all([isinstance(other, type(table)) for other in others])
#        others = _aslist(others)
#        TableClass = table.__class__
#        VariablesClass = table.variables.__class__           
#
#        others = [align_arraytables(table, other, *args, method='left', **kwargs) for other in others] 
#        datavariableitems = [data_variables(table, other, *args, **kwargs) for other in others]  
#        assert set(datavariableitems) == 1
#        datavariables = datavariableitems[0]
#        axesvariableitems = [axes_variables(table, other, *args, **kwargs) for other in others]  
#        assert set(axesvariableitems) == 1
#        axesvariables = axesvariableitems[0]
#        assert all([key not in axesvariables.keys() for key in datavariables.keys()])
#
#        xarray = function(xarray, otherxarrays, *args, *kwargs)
#        newvariables = datavariables.copy()
#        newvariables.update(axesvariables)        
#        return TableClass(data=newdataset, variables=VariablesClass(newvariables), name=kwargs.get('name', table.name))
#    update_wrapper(wrapper, function)
#    return wrapper















