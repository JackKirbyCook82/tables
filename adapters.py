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
__all__ = ['flattable_transform', 'arraytable_inversion', 'arraytable_transform', 'arraytable_operation', 'arraytable_combine', 'arraytable_layer', 'arraytable_reconcile']
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""


_aslist = lambda items: [items] if not isinstance(items, (tuple, list)) else list(items)  
_flatten = lambda nesteditems: [item for items in nesteditems for item in items]
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


def arraytable_combination(function):
    def wrapper(tables, *args, axis=None, axes=[], **kwargs):
        table, others = _aslist(tables)[0], _aslist(tables)[1:]
        if not others: return table
        assert all([isinstance(other, type(table)) for other in others])        
        TableClass = tables[0].__class__
        VariablesClass = tables[0].variables.__class__          
        
        axes = [item for item in [*_aslist(axis), *_aslist(axes)] if item is not None]
        newxarray, newvariables = function(table, others, *args, axes=axes, **kwargs)        
        try: newdataset = newxarray.to_dataset()
        except: newdataset = newxarray
    
        return TableClass(data=newdataset, variables=VariablesClass(newvariables), name=kwargs.get('name', table.name))
    update_wrapper(wrapper, function)
    return wrapper


def arraytable_combine(function):
    @arraytable_combination
    def wrapper(table, others, *args, axes, **kwargs):  
        assert table.layers == 1
        assert all([other.layers == 1 for other in others])
        assert len(_aslist(axes)) == 1  
        
        datakey, otherdatakeys = table.datakeys[0], [other.datakeys[0] for other in others]
        axis = _aslist(axes)[0]
        
        assert all([datakey == otherdatakey for otherdatakey in otherdatakeys])
        assert axis not in [datakey, *otherdatakeys]
        assert all([table.variables == other.variables for other in others])
        
        others = [align_arraytables(table, other, *args, method='left', **kwargs)[-1] for other in others]         
        dataarray, otherdataarrays = table.dataarrays[datakey], [list(table.dataarrays.values())[0] for other in others]
        newdataarray = function(dataarray, otherdataarrays, *args, **kwargs)
        newvariables = table.variables.copy()
        
        return newdataarray, newvariables
    update_wrapper(wrapper, function)
    return wrapper


def arraytable_layer(function):
    @arraytable_combination
    def wrapper(table, others, *args, axes, **kwargs):
        datakeys, otherdatakeys = table.datakeys, _flatten([other.datakeys for other in others])
        axes = _aslist(axes)          
               
        assert len(set([*datakeys, *otherdatakeys])) == len([*datakeys, *otherdatakeys])          

        for axis in axes:
            table = table.removescope(axis) if axis in table.scopekeys else table
            others = [other.removescope(axis) if axis in other.scopekeys else other for other in others]        
                
        for datakey in set([*datakeys, *otherdatakeys]):
            assert datakey not in table.headerkeys
            assert all([datakey not in other.headerkeys for other in others])       
        
        others = [align_arraytables(table, other, *args, method='outer', **kwargs)[-1] for other in others] 
        dataset, otherdatasets = table.dataset, [other.dataset for other in others]        
        newdataset = function(dataset, otherdatasets, *args, **kwargs)
        datavariables, axesvariables = {}, {}

        for other in others: 
            datavariables.update(data_variables(table, other, *args, **kwargs))
            axesvariables.update(axes_variables(table, other, *args, **kwargs))  
        newvariables = {**datavariables, **axesvariables}
                
        return newdataset, newvariables
    update_wrapper(wrapper, function)
    return wrapper


def arraytable_reconcile(function):
    @arraytable_combination
    def wrapper(table, others, *args, axes, method, **kwargs):
        assert table.layers == 1
        assert all([other.layers == 1 for other in others])        
        
        datakey, otherdatakeys = table.datakeys[0], [other.datakeys[0] for other in others]
        axes = _aslist(axes)       
        
        assert all([datakey == otherdatakey for otherdatakey in otherdatakeys])
        assert all([axis not in (datakey, *otherdatakeys) for axis in axes])
        assert all([axis not in table.headerkeys for axis in axes])
        assert all([all([axis not in other.headerkeys for other in others]) for axis in axes])

        for axis in axes:
            table = table.removescope(axis) if axis in table.scopekeys else table
            for other in others: other = other.removescope(axis) if axis in other.scopekeys else other
        
        assert all([table.variables == other.variables for other in others])
            
        others = [align_arraytables(table, other, *args, method='left', **kwargs)[-1] for other in others]                      
        dataarray, otherdataarrays = table.dataarrays[datakey], [list(table.dataarrays.values())[0] for other in others]
        newdataarray = function(dataarray, otherdataarrays, *args, method=method, **kwargs)
        newvariables = table.variables.copy()        
        
        return newdataarray, newvariables
    update_wrapper(wrapper, function)
    return wrapper        
    
   














