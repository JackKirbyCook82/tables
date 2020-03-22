# -*- coding: utf-8 -*-
"""
Created on Mon August 19 2019
@name:   Table Package Options
@author: Jack Kirby Cook

"""

import pandas as pd
import numpy as np
from scipy.linalg import cholesky, eigh
import json
from collections import OrderedDict as ODict

from tables.views import ArrayTableView, FlatTableView, HistTableView
from tables.tables import ArrayTable, FlatTable, HistTable
import tables.combinations as combinations
import tables.operations as operations
import tables.transformations as transformations
import tables.processors as processors

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['ArrayTable', 'FlatTable', 'HistTable', 'HistCollection', 'set_options', 'get_option', 'show_options', 'combinations', 'operations', 'transformations', 'processors']
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""


_OPTIONS = {'linewidth':100, 'maxrows':30, 'maxcolumns':10, 'threshold':100, 'precision':3, 'fixednotation':True, 'framechar':'='}
_PDMAPPING = {"display.max_rows":"maxrows", "display.max_columns":"maxcolumns"}
_NPMAPPING = {"linewidth":"linewidth", "threshold":"threshold", "precision":"precision", "suppress":"fixednotation"}


def apply_options():
    for key, value in _PDMAPPING.items(): pd.set_option(key, _OPTIONS[value])
    np.set_printoptions(**{key:_OPTIONS[value] for key, value in _NPMAPPING.items()})

    global ArrayTableView, ArrayTable
    global FlatTableView, FlatTable
    global HistTableView, HistTable
    ArrayTableView = ArrayTableView.factory(framechar=_OPTIONS['framechar'], framewidth=_OPTIONS['linewidth'])
    FlatTableView = FlatTableView.factory(framechar=_OPTIONS['framechar'], framewidth=_OPTIONS['linewidth'])
    HistTableView = HistTableView.factory(framechar=_OPTIONS['framechar'], framewidth=_OPTIONS['linewidth'])
    ArrayTable = ArrayTable.factory(view=ArrayTableView)
    FlatTable = FlatTable.factory(view=FlatTableView)
    HistTable = HistTable.factory(view=HistTableView)


def set_options(**kwargs):
    global _OPTIONS
    _OPTIONS.update(kwargs)
    apply_options()

    
def get_option(key): return _OPTIONS[key]
def show_options(): 
    optionstrings = json.dumps({key:str(value) for key, value in _OPTIONS.items()}, sort_keys=True, indent=3, separators=(',', ' : '))
    print('Table Options {}\n'.format(optionstrings))


class HistCollection(ODict):       
    def __init__(self, *histtables):
        assert all([isinstance(histtable, HistTable) for histtable in histtables])
        super().__init__([(histtable.name, histtable) for histtable in histtables])
        self.__correlationmatrix = np.zeros((len(self), len(self)))
        np.fill_diagonal(self.__correlationmatrix, 1)
    
    def __call__(self, size, *args, **kwargs): 
        sample_matrix = self.sample_matrix(size, *args, **kwargs)
        return sample_matrix
        
    def sample_matrix(self, size, *args, method='cholesky', **kwargs):
        sample_matrix = np.array([histtable(size) for histtable in self.values()]) 
        if method == 'cholesky':
            correlation_matrix = cholesky(self.__correlationmatrix, lower=True)
        elif method == 'eigen':
            evals, evecs = eigh(self.__correlationmatrix)
            correlation_matrix = np.dot(evecs, np.diag(np.sqrt(evals)))
        else: raise ValueError(method)
        return np.dot(correlation_matrix, sample_matrix)        


    
    
    
    
    
    
    
    
    
    
    









 

