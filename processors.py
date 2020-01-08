# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 2019
@name    Table Processors Objects
@author: Jack Kriby Cook

"""

import json
from collections import namedtuple as ntuple

from utilities.tree import Node, Tree, Renderer

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['Meta', 'Pipeline', 'Calculation', 'Renderer']
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""


_aslist = lambda items: [items] if not isinstance(items, (list, tuple)) else list(items)   


MetaSgmts = ntuple('MetaSgmts', 'universe headers scope')
class Meta(MetaSgmts):
    def todict(self): return self._asdict()   
    def __new__(cls, universe, *headers, **scope): return super().__new__(cls, universe, headers, scope)
         
    def __str__(self): 
        universe = self.universe
        headers = (self.headers[0] if len(self.headers) == 1 else ', '.join(list(self.headers))) if self.headers else None
        scope = ', '.join(['='.join([key, value]) for key, value in self.scope.items()]) if self.scope else None
        content = {key:value for key, value in {'universe':universe, 'headers':headers, 'scope':scope}.items() if value}
        return json.dumps(content, sort_keys=False, indent=3, separators=(',', ' : '), default=str)


class Pipeline(Node):
    def __init__(self, key, function, tables, parms, meta, parent=None, children=[]):
        assert isinstance(parms, dict)
        self.__function = function
        self.__tables = _aslist(tables)
        self.__parms = parms
        self.__meta = meta
        self.__table = None
        super().__init__(key, parent=parent, children=children)

    @property
    def calculated(self): return self.__table is not None 
    @property
    def table(self): return self.__table
    @property
    def tables(self): return self.__tables
    @property
    def meta(self): return self.__meta
    @property
    def parms(self): return self.__parms

    def __str__(self): 
        content = {}
        if self.__tables: content['tables'] = self.__tables
        if self.__parms: content['parms'] = self.__parms
        if self.__meta: content['meta'] = json.loads(str(self.__meta))
        return json.dumps(content, sort_keys=False, indent=3, separators=(',', ' : '), default=str)  

    def __call__(self, *args, **kwargs):
        if self.calculated: return self.__table
        tables = [child(*args, **kwargs) for child in self.children]
        print("Running Pipeline: '{}'\n".format(str(self.key)))
        self.__table = self.__function(self.key, *tables, *args, **self.__parms, **kwargs)
        self.__table.rename(self.__parms.get('name', self.key))
        return self.__table


class FrozenCalculationError(Exception): pass


class Calculation(Tree):     
    @property
    def frozen(self): return self.__frozen
    def __init__(self, *args, **kwargs):
        self.__frozen = False
        super().__init__(*args, **kwargs)
    
    def __str__(self):
        namestr = '{} ("{}")'.format(self.name if self.name else self.__class__.__name__, self.key)
        content = {key:{**json.loads(str(pipeline.meta)), 'tables':', '.join(pipeline.tables)} if pipeline.tables else json.loads(str(pipeline.meta)) for key, pipeline in self.items()}        
        jsonstr = json.dumps(content, sort_keys=False, indent=3, separators=(',', ' : '), default=str)  
        return ' '.join([namestr, jsonstr])
        
    def create(self, **kwargs):
        if self.frozen: raise FrozenCalculationError(repr(self))
        def decorator(function):
            create_pipeline = lambda key, values: Pipeline(key, function, tables=values.get('tables', []), parms=values.get('parms', {}), meta=values['meta'])
            self.update({key:create_pipeline(key, values) for key, values in kwargs.items()})
            return function
        return decorator

    def __call__(self, *args, **kwargs):
        if self.frozen: raise FrozenCalculationError(repr(self))
        for key, pipeline in self.items(): 
            self[key].addchildren(*[self[tablekey] for tablekey in pipeline.tables])
        self.__frozen = True

    
    

    