import pandas as pd
import pdblp
from typing import Optional
import os
from datetime import datetime, timedelta, date
import sgs
import numpy as np
from tqdm import tqdm
import json
import requests
import urllib.request

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import STL

def deseasonalize(var,type):
    if type == "naive":
        naive = seasonal_decompose(var.dropna(),model="additive")
        var_dessaz = var.dropna() - naive.seasonal
    
    if type == "STL":
        stl = STL(var.dropna()).fit()
        var_dessaz = var.dropna() - stl.seasonal

    return var_dessaz