# -*- coding: utf-8 -*-
"""
Created on Mon August 19 2019
@name:   Table Package Options
@author: Jack Kirby Cook

"""

import pandas as pd
import numpy as np

from tables.tables import ArrayTable, FlatTable
from tables.views import ArrayTableView, FlatTableView

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['ArrayTable', 'FlatTable']
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""


_FRAMECHAR = '='
_OPTIONS = dict(linewidth=100, maxrows=30, maxcolumns=10, threshold=100, precision=3, fixednotation=True)
_PDMAPPING = {"display.max_rows":"maxrows", "display.max_columns":"maxcolumns"}
_NPMAPPING = {"linewidth":"linewidth", "threshold":"threshold", "precision":"precision", "suppress":"fixednotation"}


def apply_options():
    for key, value in _PDMAPPING.items(): pd.set_option(key, _OPTIONS[value])
    np.set_printoptions(**{key:_OPTIONS[value] for key, value in _NPMAPPING.items()})

def set_options(*args, **kwargs):
    global _OPTIONS
    _OPTIONS.update(kwargs)
    apply_options()
    
def get_option(key): return _OPTIONS[key]
def show_options(): print('Table Options: ' + ', '.join([' = '.join([key, str(value)]) for key, value in _OPTIONS.items()]))


ArrayTableView.setframe(_FRAMECHAR, _OPTIONS['linewidth'])
FlatTableView.setframe(_FRAMECHAR, _OPTIONS['linewidth'])

ArrayTable.settableview(ArrayTableView)
FlatTable.settableview(FlatTableView)
