import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import pdb
import numpy as np
df = pd.read_csv('dataset_diabetes/diabetic_data.csv')
# patient_nbr: use for making sure we don't have same person in train/test
category_cols = ['admission_type_id','discharge_disposition_id','admission_source_id']
df.drop(['encounter_id','medical_specialty','payer_code','weight'], axis = 1, inplace = True)
df = df.replace('?', np.nan)
df = df.dropna(axis=0)
for col in list(df):
    if is_string_dtype(df[col]):
        one_hot = pd.get_dummies(df[col])
        if len(list(one_hot)) == 2: # this is binary so only need one column
            one_hot = one_hot.take([0], axis=1)
        df = pd.concat([df, one_hot], axis=1)
        df.drop(col, axis = 1, inplace = True)
    elif is_numeric_dtype(df[col]):
        if col in category_cols:
            df = pd.concat([df, pd.get_dummies(df[col])], axis=1)
            df.drop(col, axis = 1, inplace = True)
        else: # this is a count
            pass
    else:
        print('what...')
        pdb.set_trace()
print('at end')
# some patients appear multiple times, make sure to split appropriately
FF = df.groupby(['patient_nbr']).size()
GG = FF[FF != 1]
multi_patients = GG.index.values # indices of patients who appear multiple times
M = df.as_matrix()

# split dataset 80/20, tr/te
# first split multiply-occurring patients 80/20
np.random.seed(0)
nP = multi_patients.shape[0]
idx_p = np.random.permutation(nP)
training_idx_p, test_idx_p = idx_p[:int(nP*0.8)], idx_p[int(nP*0.8):]
train_patients = multi_patients[training_idx_p]
test_patients  = multi_patients[test_idx_p]

# now split remaining patients 80/20
remaining = np.setdiff1d(M[:,0],multi_patients)
nR = remaining.shape[0]
idx_r = np.random.permutation(nR)
training_idx_r, test_idx_r = idx_r[:int(nR*0.8)], idx_r[int(nR*0.8):]
train_patients_r = remaining[training_idx_r]
test_patients_r  = remaining[test_idx_r]

train_patients = np.append(train_patients, train_patients_r)
test_patients  = np.append(test_patients, test_patients_r)

# get counts
n = M.shape[0]
tr_count = 0
te_count = 0
print('get counts')
for i in range(n):
    if M[i,0] in train_patients:
        tr_count += 1
    elif M[i,0] in test_patients:
        te_count += 1
    else:
        print('oh....')
        pdb.set_trace()

d = M.shape[1]-4 # subtract y labels, and patient ID
XTR = np.zeros((tr_count,d))
YTR = np.zeros((tr_count,))
XTE = np.zeros((te_count,d))
YTE = np.zeros((te_count,))

ixr = 0
ixe = 0
for i in range(n):
    print('i=' + str(i) + ' of ' + str(n))
    x = M[i,1:-3]#.reshape(1,d)
    y = np.argmax(M[i,-3:]) # just scalar
    if M[i,0] in train_patients:
        XTR[ixr,:] = x
        YTR[ixr]   = y
        ixr += 1
    elif M[i,0] in test_patients:
        XTE[ixe,:] = x
        YTE[ixe]   = y
        #XTE = np.append(XTE, x, axis=0)
        #YTE = np.append(YTE, y)
        ixe += 1
    else:
        print('oh....')
        pdb.set_trace()

std_tr = np.std(XTR,axis=0)
std_te = np.std(XTE,axis=0)
remove_tr = np.where(std_tr == 0.0)[0]
remove_te = np.where(std_te == 0.0)[0]
remove = np.concatenate((remove_tr, remove_te))
mask = np.ones(XTR.shape[1], dtype=bool)
mask[remove] = False
XTR = XTR[:,mask]
XTE = XTE[:,mask]
DICT_TR = {}
DICT_TE = {}
DICT_P  = {}
DICT_TR['X'] = XTR
DICT_TR['Y'] = YTR
DICT_TE['X'] = XTE
DICT_TE['Y'] = YTE
DICT_TR['patients'] = train_patients
DICT_TE['patients']  = test_patients
DICT_P['multi_patients'] = multi_patients
DICT_P['remaining_patients'] = remaining
DICT_P['mask'] = mask
name = 'diabetes'
np.savez_compressed(name + '_train', **DICT_TR)
np.savez_compressed(name + '_test',  **DICT_TE)
np.savez_compressed(name + '_info',  **DICT_P)
pdb.set_trace()
#df.to_csv('dataset_diabetes/diabetic_data_cleaned.csv')
print('done')
