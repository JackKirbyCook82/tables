# -*- coding: utf-8 -*-
"""
Created on Tues Aug 20 2019
@name:   Table Adapters Decorators
@author: Jack Kirby Cook

"""

from functools import update_wrapper
from collections import OrderedDict as ODict

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
           
        dataframe, variables = table.dataframe, table.variables.copy()
        newdataframe, newvariables = function(self, dataframe, variables=variables, *args, **kwargs)
        variables = variables.update(ODict([(key, value) for key, value in newvariables.items()]))        
        
        return TableClass(data=newdataframe, variables=variables, name=kwargs.get('name', table.name))    
    update_wrapper(wrapper, function)
    return wrapper


def arraytable_inversion(function):
    @retag
    def wrapper(self, table, *args, **kwargs):
        assert table.layers == 1
        TableClass = table.__class__      
                
        dataarray, variables = list(table.dataarrays.values())[0], table.variables.copy()
        newdataarray, newvariables = function(self, dataarray, *args, variables=variables, **kwargs)
        newdataset = newdataarray.to_dataset()
        variables = variables.update(ODict([(key, value) for key, value in newvariables.items()]))
        
        return TableClass(data=newdataset, variables=variables, name=kwargs.get('name', table.name))
    update_wrapper(wrapper, function)
    return wrapper


def arraytable_transform(function):
    @retag
    def wrapper(self, table, *args, axis, **kwargs):
        TableClass = table.__class__     
        
        dataset, variables = table.dataset, table.variables.copy()
        newdataset, newvariables = function(self, dataset, *args, axis=axis, variables=variables, **kwargs)
        variables = variables.update(ODict([(key, value) for key, value in newvariables.items()]))   
        
        return TableClass(data=newdataset, variables=variables, name=kwargs.get('name', table.name))
    update_wrapper(wrapper, function)
    return wrapper


def arraytable_operation(function):
    @retag
    def wrapper(table, other, *args, axes=[], axis=None, noncoreaxes=[], noncoreaxis=None, **kwargs):
        assert isinstance(other, type(table))
        assert table.layers == other.layers == 1
        TableClass = table.__class__
        
        noncoreaxes = [noncoreaxis for noncoreaxis in [*noncoreaxes, noncoreaxis] if noncoreaxis is not None]
        axes = [axis for axis in [*axes, axis] if axis is not None]
        datakey, otherdatakey = table.datakeys[0], other.datakeys[0]
        noncoreaxes = list(set([*noncoreaxes, datakey, otherdatakey]))

        for axis in [*axes, *noncoreaxes]:
            try: table = table.squeeze(axis)
            except: pass
            try: other = other.squeeze(axis)
            except: pass

        for noncoreaxis in noncoreaxes:
            table = table.removescope(noncoreaxis) if noncoreaxis in table.scopekeys else table
            other = other.removescope(noncoreaxis) if noncoreaxis in other.scopekeys else other
  
        datavariables = data_variables(table, other, *args, **kwargs)
        variables = axes_variables(table, other, *args, **kwargs)  

        dataarray, otherdataarray = table.dataarrays[datakey], other.dataarrays[otherdatakey]
        newdataarray, newvariables = function(dataarray, otherdataarray, *args, variables=datavariables, **kwargs)  
        variables = variables.update(ODict([(key, value) for key, value in newvariables.items()]))       
        newdataset = newdataarray.to_dataset()  
        
        return TableClass(data=newdataset, variables=variables, name=kwargs.get('name', table.name))
    update_wrapper(wrapper, function)
    return wrapper        
        

def arraytable_combination(function):
    def wrapper(tables, *args, axis=None, axes=[], noncoreaxis=None, noncoreaxes=[], **kwargs):
        table, others = _aslist(tables)[0], _aslist(tables)[1:]
        if not others: return table
        assert all([isinstance(other, type(table)) for other in others])        
        TableClass = tables[0].__class__
        tablename = table.name
        VariablesClass = tables[0].variables.__class__  
        variablesname = tables[0].variables.name        

        axes = [item for item in [*_aslist(axis), *_aslist(axes)] if item is not None]
        noncoreaxes = [item for item in [*_aslist(noncoreaxis), *_aslist(noncoreaxes)] if item is not None]
        
        for noncoreaxis in noncoreaxes:
            try: table = table.squeeze(noncoreaxis)
            except: pass
            for i in range(len(others)): 
                try: others[i] = others[i].squeeze(noncoreaxis)
                except: pass
        
        for noncoreaxis in noncoreaxes:
            table = table.removescope(noncoreaxis) if noncoreaxis in table.scopekeys else table
            others = [other.removescope(noncoreaxis) if noncoreaxis in other.scopekeys else other for other in others]        
        
        newxarray, newvariables = function(table, others, *args, axes=axes, **kwargs)        
        try: newdataset = newxarray.to_dataset()
        except: newdataset = newxarray
    
        return TableClass(data=newdataset, variables=VariablesClass(newvariables, name=variablesname), name=kwargs.get('name', tablename))
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
        assert all([dict(table.variables) == dict(other.variables) for other in others])
        
        others = [align_arraytables(table, other, *args, method='outer', **kwargs)[-1] for other in others]         
        dataarray, otherdataarrays = table.dataarrays[datakey], [list(other.dataarrays.values())[0] for other in others]
        newdataarray = function(dataarray, otherdataarrays, *args, axis=axis, **kwargs)
        newvariables = table.variables.copy()
        
        return newdataarray, newvariables
    update_wrapper(wrapper, function)
    return wrapper


def arraytable_layer(function):
    @arraytable_combination
    def wrapper(table, others, *args, **kwargs):
        datakeys, otherdatakeys = table.datakeys, _flatten([other.datakeys for other in others])         
        assert len(set([*datakeys, *otherdatakeys])) == len([*datakeys, *otherdatakeys])          

        for datakey in set([*datakeys, *otherdatakeys]):
            assert datakey not in table.headerkeys
            assert all([datakey not in other.headerkeys for other in others])       
        
        others = [align_arraytables(table, other, *args, method='outer', **kwargs)[-1] for other in others] 
        dataset, otherdatasets = table.dataset, [other.dataset for other in others]        
        newdataset = function(dataset, otherdatasets, *args, **kwargs)
        
        datavariables, axesvariables = {}, {}
        for other in others: 
            datavariables.update(**data_variables(table, other, *args, **kwargs))
            axesvariables.update(**axes_variables(table, other, *args, **kwargs))  
        newvariables = {**datavariables, **axesvariables}
                
        return newdataset, newvariables
    update_wrapper(wrapper, function)
    return wrapper
















