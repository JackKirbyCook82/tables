# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 2019
@name    Calculation Objects
@author: Jack Kriby Cook

"""

from functools import update_wrapper

import tables as tbls

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['Calculation']
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""


class Calculation(dict):
    def __new__(cls, *args, webapi, variables, **kwargs):
        def decorator(function): 
            def wrapper(*wargs, **wkwargs): return function(*wargs, **wkwargs)
            update_wrapper(wrapper, function)  
            instance = super(Calculation, cls).__new__(cls, *args, **kwargs)
            setattr(instance, 'function', wrapper)
            setattr(instance, 'webapi', webapi)
            setattr(instance, 'variables', variables)            
            return instance
        return decorator

    def __call__(self, *args, **kwargs):
        self.webapi.reset()
        tables = {key:value(*args, **kwargs) for key, value in self.items()}
        return self.function(*list(tables.values()), *args, **kwargs)
    
    def register(self, tablekey, *args, name, **kwargs): 
        def decorator(function):
            def wrapper(*wargs, **wkwargs):
                self.setwebapi(*args, **kwargs)                                                       
                dataframe = self.webapi(*wargs, **wkwargs)
                flattable = tbls.FlatTable(data=dataframe, variables=self.variables, name=name)
                arraytable = flattable.unflatten(*self.tablekeys(*args, **kwargs))
                for axis in arraytable.headerkeys: arraytable = arraytable.sort(axis, ascending=True)
                return function(arraytable, *wargs, **wkwargs)            
            
            update_wrapper(wrapper, function)        
            self[tablekey] = wrapper
            return wrapper
        return decorator
    
    def setwebapi(self, *args, universe, index, header,  scope={}, **kwargs):
        self.webapi.reset()
        self.webapi.setitems(universe=universe, index=index, header=header, **scope)
        print(str(self.webapi), '\n')        
    
    def tablekeys(self, *args, universe, index, header, headers=[], scope={}, **kwargs):
        datakeys = [universe]
        headerkeys = [index, header, *headers]
        scopekeys = list(scope.keys())        
        return [datakeys, headerkeys, scopekeys]
    
    

    
    
    
    