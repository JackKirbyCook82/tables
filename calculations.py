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


class Calculation(list):
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
        tables = [function(*args, **kwargs) for function in self]
        return self.function(*tables, *args, **kwargs)
    
    def register(self, *args, **kwargs): 
        def decorator(function):
            def wrapper(*wargs, **wkwargs):
                self.setwebapi(*args, **kwargs)                                                       
                dataframe = self.webapi(*wargs, **wkwargs)
                flattable = tbls.FlatTable(data=dataframe, variables=self.variables)
                arraytable = flattable.unflatten(*self.tablekeys(*args, **kwargs))
                for axis in arraytable.headerkeys: arraytable = arraytable.sort(axis, ascending=True)
                return function(arraytable, *wargs, **wkwargs)            
            
            update_wrapper(wrapper, function)        
            self.append(wrapper)
            return wrapper
        return decorator
    
    def setwebapi(self, universe, index, header, *args, scope={}, **kwargs):
        self.webapi.reset()
        self.webapi.setitems(universe=universe, index=index, header=header, **scope)
        print(str(self.webapi), '\n')        
    
    def tablekeys(self, universe, index, header, *args, headers=[], scope={}, **kwargs):
        datakeys = [universe]
        headerkeys = [index, header, *headers]
        scopekeys = list(scope.keys())        
        return [datakeys, headerkeys, scopekeys]
    

    
    
    