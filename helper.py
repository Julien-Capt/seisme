# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 16:24:50 2021

@author: JuHik
"""

from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def normalization(x: np.ndarray):
    
    mean = x.mean(axis=0)
    std = x.std(axis=0)

    return  (x - mean) / std