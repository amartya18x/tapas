import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import pdb
import numpy as np
df = pd.read_csv('wdbc.data',header=None)
df[1] = df[1].map({'B':0, 'M':1}) # change label to 0/1
# split dataset 70/30, tr/te
# first split multiply-occurring patients 70/30
n = df.shape[0]
np.random.seed(0)
idx = np.random.permutation(n)
training_idx, test_idx = idx[:int(n*0.7)], idx[int(n*0.7):]
XTR = df.loc[training_idx,2:].as_matrix()
YTR = df.loc[training_idx,1].as_matrix()
XTE = df.loc[test_idx,2:].as_matrix()
YTE = df.loc[test_idx,1].as_matrix()
DICT_TR = {}
DICT_TE = {}
DICT_P  = {}
DICT_TR['X'] = XTR
DICT_TE['X'] = XTE
DICT_TR['Y'] = YTR
DICT_TE['Y'] = YTE
DICT_P['training_idx'] = training_idx
DICT_P['test_idx'] = test_idx
name = 'cancer'
np.savez_compressed(name + '_train', **DICT_TR)
np.savez_compressed(name + '_test',  **DICT_TE)
np.savez_compressed(name + '_info',  **DICT_P)
pdb.set_trace()
