# -*- coding: utf-8 -*-
"""
Created on Mon August 19 2019
@name:   Table Package Options
@author: Jack Kirby Cook

"""

import pandas as pd
import numpy as np
import json

import tables.tables as tbls
import tables.views as views

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['ArrayTable', 'FlatTable', 'set_options', 'get_option', 'show_options']
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""


_OPTIONS = {'linewidth':100, 'maxrows':30, 'maxcolumns':10, 'threshold':100, 'precision':3, 'fixednotation':True, 'framechar':'='}
_PDMAPPING = {"display.max_rows":"maxrows", "display.max_columns":"maxcolumns"}
_NPMAPPING = {"linewidth":"linewidth", "threshold":"threshold", "precision":"precision", "suppress":"fixednotation"}


def apply_options():
    for key, value in _PDMAPPING.items(): pd.set_option(key, _OPTIONS[value])
    np.set_printoptions(**{key:_OPTIONS[value] for key, value in _NPMAPPING.items()})


def set_options(**kwargs):
    global ArrayTable, FlatTable
    global ArrayTableView, FlatTableView
    global _OPTIONS
    _OPTIONS.update(kwargs)
    ArrayTable = ArrayTable.factory(ArrayTableView.factory(framechar=_OPTIONS['framechar'], framelength=_OPTIONS['linewidth']))
    FlatTable = FlatTable.factory(FlatTableView.factory(framechar=_OPTIONS['framechar'], framelength=_OPTIONS['linewidth']))
    apply_options()

    
def get_option(key): return _OPTIONS[key]
def show_options(): 
    optionstrings = json.dumps({key:str(value) for key, value in _OPTIONS.items()}, sort_keys=True, indent=3, separators=(',', ' : '))
    print('Table Options {}\n'.format(optionstrings))


ArrayTableView = views.ArrayTableView.factory(framechar=_OPTIONS['framechar'], framelength=_OPTIONS['linewidth'])
ArrayTable = tbls.ArrayTable.factory(view=ArrayTableView)

FlatTableView = views.FlatTableView.factory(framechar=_OPTIONS['framechar'], framelength=_OPTIONS['linewidth'])
FlatTable = tbls.FlatTable.factory(view=FlatTableView)














 

