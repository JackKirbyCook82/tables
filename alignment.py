# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 2019
@name:   Table Alignment Functions
@author: Jack

"""

import xarray as xr

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['align_variables', 'align_arraytables']
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""


_aslist = lambda items: [items] if not isinstance(items, (tuple, list)) else list(items)  


def align_variables(variables, others):
        variables, othervariables = variables.copy(), others.copy()     
        for key in set([*variables.keys(), *othervariables.keys()]): 
            if all([key in variables.keys(), key in othervariables.keys()]): assert variables[key] == othervariables[key]        
        variables.update(othervariables)
        return variables


def align_arraytables(table, other, *args, method, noncoreaxes=[], **kwargs):
    assert isinstance(other, type(table))
    assert method in ('outer', 'inner', 'left', 'right', 'exact')
    dataset, otherdataset = xr.align(table.dataset, other.dataset, join=method, copy=True, exclude=_aslist(noncoreaxes))
    variables = align_variables(table.variables, other.variables)
    table = table.__class__(data=dataset, variables=variables, name=table.name).sortall(ascending=True)
    other = other.__class__(data=otherdataset, variables=variables, name=other.name).sortall(ascending=True)
    return table, other