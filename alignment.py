# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 2019
@name:   Table Alignment Functions
@author: Jack

"""

import xarray as xr
from functools import update_wrapper

from utilities.dispatchers import keyword_singledispatcher as keydispatcher

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['align_arraytables', 'align_variables', 'axes_variables', 'data_variables']
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""


_aslist = lambda items: [items] if not isinstance(items, (tuple, list)) else list(items)  
_union = lambda x, y: list(set(x) | set(y))
_intersection = lambda x, y: list(set(x) & set(y))


class AlignmentError(Exception): pass
class VariableAlignmentError(AlignmentError): pass
class DatasetAlignmentError(AlignmentError): pass
    

def align_arraytables(table, other, *args, method, noncoreaxes=[], **kwargs):
    assert isinstance(other, type(table))    
    assert method in ('outer', 'inner', 'left', 'right', 'exact')

    try: dataset, otherdataset = xr.align(table.dataset, other.dataset, join=method, copy=True, exclude=_aslist(noncoreaxes))
    except ValueError: raise DatasetAlignmentError()
    variables, othervariables = align_variables(table.variables, other.variables, *args, method=method, noncoreaxes=noncoreaxes, **kwargs)

    table = table.__class__(data=dataset, variables=variables, name=table.name).sortall(ascending=True)
    other = other.__class__(data=otherdataset, variables=othervariables, name=other.name).sortall(ascending=True)
    return table, other


def _align_variables(function):
    def wrapper(variables, others, *args, noncoreaxes=[], **kwargs):
        assert isinstance(others, type(variables))
        VariablesClass = variables.__class__
            
        variables, others = variables.copy(), others.copy()
        noncorevariables = {key:variables.pop(key) for key in _aslist(noncoreaxes) if key in variables.keys()}    
        noncoreothers = {key:others.pop(key) for key in _aslist(noncoreaxes) if key in others.keys()}          
        variables, other = function(variables, others, *args, **kwargs)
        variables.update(noncorevariables)
        others.update(noncoreothers)
        return VariablesClass(variables), VariablesClass(others)
    update_wrapper(wrapper, function)
    return wrapper


@keydispatcher('method')
def align_variables(variables, others, *args, method, **kwargs): raise KeyError(method)

@align_variables.register('outer')
@_align_variables
def align_variables_outer(variables, others, *args, **kwargs):
    variables = {key:(variables[key] if key in variables.keys() else others[key]) for key in _union(variables.keys(), others.keys())}
    others = {key:(others[key] if key in others.keys() else others[key]) for key in _union(variables.keys(), others.keys())}
    for key in _union(variables.keys(), others.keys()): assert variables[key] == others[key]
    return variables, others

@align_variables.register('inner')
@_align_variables
def align_variables_inner(variables, others, *args, **kwargs):
    variables = {key:(variables[key] if key in variables.keys() else others[key]) for key in _intersection(variables.keys(), others.keys())}
    others = {key:(others[key] if key in others.keys() else others[key]) for key in _intersection(variables.keys(), others.keys())}
    for key in _intersection(variables.keys(), others.keys()): assert variables[key] == others[key]       
    return variables, others

@align_variables.register('left')
@_align_variables
def align_variables_left(variables, others, *args, **kwargs):
    others = {key:(others[key] if key in others.keys() else variables[key]) for key in variables.keys()}
    for key in others.keys(): assert others[key] == variables[key]
    return variables, others

@align_variables.register('right')
@_align_variables
def align_variables_right(variables, others, *args, **kwargs):    
    variables = {key:(variables[key] if key in variables.keys() else others[key]) for key in others.keys()}
    for key in variables.keys(): assert variables[key] == others[key]    
    return variables, others

@align_variables.register('exact')
@_align_variables
def align_variables_exact(variables, others, *args, **kwargs):
    assert isinstance(others, type(variables))
    if set(variables.keys()) != set(others.keys()): raise VariableAlignmentError()
    if not all([variables[key] == others[key] for key in set(variables.keys())]): raise VariableAlignmentError()        
    return variables, others


def axes_variables(table, other, *args, **kwargs):
    VariablesClass = table.variables.__class__
    variables = {key:value for key, value in table.variables.items() if key not in table.datakeys}
    othervariables = {key:value for key, value in other.variables.items() if key not in other.datakeys}
     
    for key in set([*variables.keys(), *othervariables.keys()]): 
        if all([key in variables.keys(), key in othervariables.keys()]): assert variables[key] == othervariables[key]                
    variables.update(othervariables)
    return VariablesClass(variables)


def data_variables(table, other, *args, **kwargs):
    VariablesClass = table.variables.__class__
    variables = {key:value for key, value in table.variables.items() if key in table.datakeys}
    othervariables = {key:value for key, value in other.variables.items() if key in other.datakeys}
     
    for key in set([*variables.keys(), *othervariables.keys()]): 
        if all([key in variables.keys(), key in othervariables.keys()]): assert variables[key] == othervariables[key]                
    variables.update(othervariables)
    return VariablesClass(variables)    

















