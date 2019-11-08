# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 2019
@name    Table Processors Objects
@author: Jack Kriby Cook

"""

import json
import pandas as pd
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
    def todict(self): return dict(meta=self.__meta.todict(), parms=self.__parms, tables=tuple([child.key for child in self.children]))     
    def __init__(self, key, function, parms, meta, parent=None, children=[]):
        assert isinstance(parms, dict)
        self.__function = function
        self.__parms = parms
        self.__meta = meta
        self.__table = None
        super().__init__(key, parent=parent, children=children)

    @property
    def calculated(self): return self.__table is not None 
    @property
    def table(self): return self.__table
    @property
    def meta(self): return self.__meta
    @property
    def parms(self): return self.__parms

    def __str__(self): 
        content = {'meta':json.loads(str(self.__meta))}
        if self.children: content['tables'] = tuple([child.key for child in self.children])
        if self.__parms: content ['parms'] = self.__parms
        return json.dumps(content, sort_keys=False, indent=3, separators=(',', ' : '), default=str)  

    def __call__(self, *args, **kwargs):
        if self.calculated: return self.__table
        tables = [child(*args, **kwargs) for child in self.children]
        print("Running Pipeline: '{}'\n".format(str(self.key)))
        self.__table = self.__function(self.key, *tables, *args, **self.__parms, **kwargs)
        self.__table.rename(self.__parms.get('name', self.key))
        return self.__table


class EmptyCalculationQueryError(Exception): pass
class CalculationPipelineNotConnectedError(Exception): pass
class CalculationPipelineNotCreatedError(Exception): pass


class Calculation(Tree):
    def todict(self): return {key:pipeline.todict() for key, pipeline in self.nodes.items()}
    def __init__(self, *args, **kwargs):
        self.__queue = {}
        super().__init__(*args, **kwargs)
        self.__calculations = {self.key:self}       
        
    def queue(self, key, values): self.__queue[key] = _aslist(values)
    def reset(self): self.__queue = {}
            
    def __str__(self):
        namestr = '{} ("{}")'.format(self.name if self.name else self.__class__.__name__, self.key)
        content = {key:{**json.loads(str(value.meta)), 'tables':', '.join([child.key for child in value.children])} if value.children else json.loads(str(value.meta)) for key, value in self.nodes.items()}        
        jsonstr = json.dumps(content, sort_keys=False, indent=3, separators=(',', ' : '), default=str)  
        return ' '.join([namestr, jsonstr])
        
    def create(self, **kwargs):
        def decorator(function):
            create_pipeline = lambda key, values: Pipeline(key, function, parms=values.get('parms', {}), meta=values['meta'])
            create_queue = lambda key, values: [item if isinstance(item, tuple) else (self.key, item) for item in _aslist(values.get('tables', []))]
            for key, values in kwargs.items():
                self.append(create_pipeline(key, values))
                self.queue(key, create_queue(key, values))           
            return function
        return decorator
         
    def __getitem__(self, key): 
        if key not in self.nodes.keys(): raise CalculationPipelineNotCreatedError(self.key, key)
        if key in self.__queue.keys(): raise CalculationPipelineNotConnectedError(self.key, key)
        return super().__getitem__(key) 
    
    def __call__(self, *args, **kwargs):
        for key, values in self.__queue.items():
            self.nodes[key].addchildren(*[self.__calculations[calckey].nodes[nodekey] for calckey, nodekey in values])
        self.reset()

    def extend(self, other):
        assert isinstance(other, type(self))
        assert other.key not in self.__calculations.keys()
        assert other.key != self.key
        self.__calculations[other.key] = other

    def __createdataframe(self):
        create_content = lambda calculation: {(calculation.key, key):{**value['meta'],'tables':value['tables']} for key, value in calculation.todict().items()}
        content = {}
        for calculation in self.__calculations.values(): content.update(create_content(calculation))
        return pd.DataFrame(content).transpose()        
    
    def __querydataframe(self, dataframe, universe, headers, scope):
        assert isinstance(universe, (type(None), str))
        assert isinstance(headers, tuple)
        assert isinstance(scope, dict)
        if universe: dataframe = dataframe[dataframe['universe']==universe]
        if dataframe.empty: raise EmptyCalculationQueryError() 
        for header in headers: 
            if header: dataframe[header in dataframe['headers']]
        if dataframe.empty: raise EmptyCalculationQueryError()
        for key, value in scope.items():
            if value: dataframe = dataframe[dataframe['scope'].apply(lambda x: x[key] == value if key in x.keys() else False)]
        if dataframe.empty: raise EmptyCalculationQueryError()
        dataframe = dataframe.dropna(axis=0, how='all')
        dataframe.columns.name = '{} Query'.format(self.name if self.name else 'Calculation')
        try: return dataframe.to_frame()
        except: return dataframe        
    
    def __formatdataframe(self, dataframe):
        dataframe['headers'] = dataframe['headers'].apply(lambda x: ', '.join(list(x)))
        dataframe['scope'] = dataframe['scope'].apply(lambda x: ', '.join(['='.join([key, value]) for key, value in x.items()]))
        dataframe['tables'] = dataframe['tables'].apply(lambda x: ', '.join(list(x)))
        dataframe = dataframe[['universe', 'headers', 'scope', 'tables']]
        return dataframe
    
    def query(self, *headers, universe=None, **scope): 
        dataframe = self.__createdataframe()
        dataframe = self.__querydataframe(dataframe, universe, headers, scope)
        dataframe = self.__formatdataframe(dataframe)
        return dataframe
    
    
    
    

    