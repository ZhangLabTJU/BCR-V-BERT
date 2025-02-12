import os
import sys
import re
import pandas as pd
import numpy as np

from typing import *
import warnings
warnings.filterwarnings('ignore')

path_work = '/data1/xqh/antibody/anti_bert/Vbert'
path_data = os.path.join(path_work,'pretrain_data')
path_utils = os.path.join('utils') # utils directory
sys.path.append(path_utils)
os.chdir(path_work)

