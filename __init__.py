# -*- coding: utf-8 -*-
"""
Created on Mon August 19 2019
@name:   Tables Module
@author: Jack Kirby Cook

"""

import pandas as pd
import numpy as np
import json

from tables.views import ArrayTableView, FlatTableView, HistTableView, CurveTableView
from tables.tables import ArrayTable, FlatTable, HistTable, CurveTable
import tables.combinations as combinations
import tables.operations as operations
import tables.transformations as transformations
import tables.processors as processors

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['ArrayTable', 'FlatTable', 'HistTable', 'set_options', 'get_option', 'show_options', 'combinations', 'operations', 'transformations', 'processors']
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
    global CurveTableView, CurveTable
    ArrayTableView = ArrayTableView.factory(framechar=_OPTIONS['framechar'], framewidth=_OPTIONS['linewidth'])
    FlatTableView = FlatTableView.factory(framechar=_OPTIONS['framechar'], framewidth=_OPTIONS['linewidth'])
    HistTableView = HistTableView.factory(framechar=_OPTIONS['framechar'], framewidth=_OPTIONS['linewidth'])
    CurveTableView = CurveTableView.factory(framechar=_OPTIONS['framechar'], framewidth=_OPTIONS['linewidth'])
    ArrayTable = ArrayTable.factory(view=ArrayTableView)
    FlatTable = FlatTable.factory(view=FlatTableView)
    HistTable = HistTable.factory(view=HistTableView)
    CurveTable = CurveTable.factory(view=CurveTableView)


def set_options(**kwargs):
    global _OPTIONS
    _OPTIONS.update(kwargs)
    apply_options()

    
def get_option(key): return _OPTIONS[key]
def show_options(): 
    optionstrings = json.dumps({key:str(value) for key, value in _OPTIONS.items()}, sort_keys=True, indent=3, separators=(',', ' : '))
    print('Table Options {}\n'.format(optionstrings))



    
    
    
    
    
    
    
    









 

