# -*- coding: utf-8 -*-
"""
Created on Sun Jun 2 2019
@name    Operation Functions
@author: Jack Kriby Cook

"""

from functools import update_wrapper
import xarray as xr

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['add', 'subtract', 'multiply', 'divide', 'concat']
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""


_aslist = lambda items: [items] if not isinstance(items, (list, tuple)) else list(items)     


def operationloop(function):
    def wrapper(table, *others, **kwargs):
        assert all([isinstance(other, type(table)) for other in others])    
        newtable = table
        for other in others: newtable = function(newtable, other, **kwargs)
        return newtable
    update_wrapper(wrapper, function)
    return wrapper


def operation(namefunction = lambda name, other: '&'.join([name, other]) if name != other else name):
    def decorator(dataarrayfunction):
        @operationloop
        def wrapper(table, other, *args, axes=[], **kwargs):
            assert isinstance(other, type(table))
            assert table.layers == other.layers == 1          
            TableClass = table.__class__

            datakey, otherdatakey = table.datakeys[0], other.datakeys[0]            
            dataarray, otherdataarray = table.dataarrays[datakey], other.dataarrays[otherdatakey]
            variables, othervariables = table.variables, other.variables
            
            dataarray.attrs.pop(otherdatakey, None), otherdataarray.attrs.pop(datakey, None)            
            if datakey != otherdatakey: variables.pop(otherdatakey, None), othervariables.pop(datakey, None)
            datavariable, otherdatavariable = variables.pop(datakey), othervariables.pop(otherdatakey)            
            
            method = dataarrayfunction.__name__
            axes = _aslist(axes)
            
            coredims = [dim for dim in set([*dataarray.dims, *otherdataarray.dims]) if dim not in axes]
            coreattrs = [attr for attr in set([*dataarray.attrs.keys(), *otherdataarray.attrs.keys()]) if attr not in axes]
            coreaxes = set([*coredims, *coreattrs])

            assert all([coredim in dataarray.dims and coredim in otherdataarray.dims for coredim in coredims])
            assert all([coreattr in dataarray.attrs.keys() and coreattr in otherdataarray.attrs.keys() for coreattr in coreattrs])
            for coredim in coredims: assert all([item == otheritem for item, otheritem in zip(dataarray.coords[coredim], otherdataarray.coords[coredim])])
            for coreattr in coreattrs: assert dataarray.attrs[coreattr] == otherdataarray.attrs[coreattr]  
            for coreaxis in coreaxes: assert all([variables[coreaxis] == othervariables[coreaxis]])
                       
            newname = kwargs.get('name', namefunction(table.name, other.name))     
            newvariable = datavariable.operation(otherdatavariable, *args, method=method, **kwargs)  
            newattrs = {key:value for key, value in dataarray.attrs.items() if key not in axes}
            newvariables = variables.copy()           
            
            for axis in axes:
                if all([axis in dataarray.dims, axis in otherdataarray.dims]): 
                    if len(otherdataarray.coords[axis]) == 1: otherdataarray = otherdataarray.squeeze(dim=axis)
                    elif len(dataarray.coords[axis]) == 1: dataarray = dataarray.squeeze(dim=axis)
                    else: raise ValueError(axis) 
                elif all([axis in dataarray.dims, axis in otherdataarray.attrs.keys()]): otherdataarray.attrs.pop(axis)  
                elif all([axis in dataarray.attrs.keys(), axis in otherdataarray.dims]): dataarray.attrs.pop(axis)
                elif all([axis in dataarray.attrs.keys(), axis in otherdataarray.attrs.keys()]): 
                    newattrs[axis] = getattr(variables[axis].fromstr(dataarray.attrs.pop(axis)), method)(othervariables[axis].fromstr(otherdataarray.attrs.pop(axis)))
                else: raise ValueError(axis)

            newdataarray = dataarrayfunction(dataarray, otherdataarray, *args, variables=variables, **kwargs)                        
            newvariables[newdataarray.name] = newvariable
            newdataset = newdataarray.to_dataset()
            newdataset.attrs = newattrs                        
            return TableClass(data=newdataset, variables=newvariables, name=newname)            
       
        update_wrapper(wrapper, dataarrayfunction)
        return wrapper
    return decorator


@operation(namefunction = lambda name, other: '+'.join([name, other]) if name != other else name)
def add(dataarray, other, *args, **kwargs):    
    newdataarray = dataarray + other    
    newdataarray.name = '+'.join([dataarray.name, other.name]) if dataarray.name != other.name else dataarray.name
    return newdataarray


@operation(namefunction = lambda name, other: '-'.join([name, other]) if name != other else name)
def subtract(dataarray, other, *args, **kwargs):
    newdataarray = dataarray - other    
    newdataarray.name = '-'.join([dataarray.name, other.name]) if dataarray.name != other.name else dataarray.name
    return newdataarray


@operation(namefunction = lambda name, other: '*'.join([name, other]) if name != other else name)
def multiply(dataarray, other, *args, **kwargs):
    newdataarray = dataarray * other    
    newdataarray.name = '*'.join([dataarray.name, other.name]) if dataarray.name != other.name else dataarray.name
    return newdataarray


@operation(namefunction = lambda name, other: '/'.join([name, other]) if name != other else name)
def divide(dataarray, other, *args, **kwargs):
    newdataarray = dataarray / other    
    newdataarray.name = '/'.join([dataarray.name, other.name]) if dataarray.name != other.name else dataarray.name
    return newdataarray


def concat(table, other, *args, axis, **kwargs):
    assert table.layers == other.layers == 1          
    TableClass = table.__class__

    datakey, otherdatakey = table.datakeys[0], other.datakeys[0]            
    dataarray, otherdataarray = table.dataarrays[datakey], other.dataarrays[otherdatakey]        
    variables, othervariables = table.variables, other.variables
    
    try: dataarray = dataarray.expand(axis)
    except: pass     
    try: other = other.expand(axis)
    except: pass
    assert all([axis in dataarray.dims, axis in other.dims])
    
    assert dataarray.dims == otherdataarray.dims
    assert all([dataarray.coords[key] == otherdataarray.coords[key] for key in set([*dataarray.dims, *otherdataarray.dims]) if key != axis])
    assert dataarray.attrs.keys() == otherdataarray.attrs.keys()            
    assert all([dataarray.attrs[key] == otherdataarray.attrs[key] for key in set([*dataarray.attrs.keys(), *otherdataarray.attrs.keys()]) if key != axis])            
    assert variables == othervariables

    newname = kwargs.get('name', '&'.join([table.name, other.name]) if table.name != other.name else table.name)   
    newdatakey = '&'.join([datakey, otherdatakey]) if datakey != otherdatakey else datakey 
    newattrs = {key:value for key, value in dataarray.attrs.items() if key != axis}
    newdataarray = xr.concat([dataarray, otherdataarray], dim=axis)           

    newdataset = newdataarray.to_dataset(name=newdatakey)
    newdataset.attrs = newattrs
    newvariables = variables.copy()
    return TableClass(data=newdataset, variables=newvariables, name=newname)       

@operationloop
def layer(table, other, *args, name, **kwargs):
    assert isinstance(other, type(table))
    assert other.layers == 1          
    TableClass = table.__class__

    datakeys, otherdatakeys = table.datakeys, other.datakeys
    dataset, otherdataset = table.dataset, other.dataset
    variables, othervariables = table.variables, other.variables  

    for datakey in datakeys: otherdataset.attrs.pop(datakey, None) 
    for otherdatakey in otherdatakeys: dataset.attrs.pop(otherdatakey, None)

    assert all([otherdatakey not in datakeys for otherdatakey in otherdatakeys])
    assert dataset.attrs == otherdataset.attrs

    for datakey in datakeys: othervariables.pop(datakey, None)
    for otherdatakey in otherdatakeys: variables.pop(otherdatakey, None)
      
    for dim in set([*dataset.dims, *otherdataset.dims]): 
        if all([dim in dataset.dims, dim in otherdataset.dims]):
            assert variables[dim] == othervariables[dim]
    for key in set([*dataset.attrs.keys(), *otherdataset.attrs.keys()]):
        assert dataset.attrs[key] == otherdataset.attrs[key]
        assert variables[key] == othervariables[key]

    newvariables = variables.copy()
    newvariables = newvariables.update(othervariables)
    newdataset = xr.merge([dataset, otherdataset])  
    newdataset.attrs = dataset.attrs
    return TableClass(data=newdataset, variables=newvariables, name=name)


#def reconcile(table, other, *args, name, **kwargs):
#    assert isinstance(other, type(table))
#    assert table.layers == other.layers
#    TableClass = table.__class__
#    
#    
#    
#    return TableClass(data=, variables=, name=name)

















