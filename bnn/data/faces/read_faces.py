import os
from glob import glob
from PIL import Image
import pdb
import numpy as np
import pandas as pd

files = []
names = []
#start_dir = os.getcwd()
pattern   = "*.jpg"
prefix = 'lfwa'
start_dir = 'lfwa/lfw2'
male = []
female = []
for line in open(prefix + '/male_names.txt','r'):
    line = line.rstrip()
    male.append(line)
male.append('Yasser_Arafat_0005.jpg')
male.append('Robert_Evans_0001.jpg')
male.append('Robert_Evans_0002.jpg')
for line in open(prefix + '/female_names.txt','r'):
    line = line.rstrip()
    female.append(line)
female.append('Tara_Kirk_0001.jpg')
dirs = []
for dir,dn,fn in os.walk(start_dir):
    files.extend(glob(os.path.join(dir,pattern))) 
    names.extend(fn)
    dirs.extend(dn)

m = len(dirs)
people_ix = []
ix_people = {}
count = 0
for n in names:
    ix = dirs.index(n[:-9])
    people_ix.append(ix)
    if ix not in ix_people:
        ix_people[ix] = [count]
    else:
        ix_people[ix].append(count)
    count = count + 1

n = len(files)
X = np.zeros((n,1,50,50))
Y = np.zeros((n,),dtype=int)
count = 0
for f in files:
    img = Image.open(f)
    img = img.resize((50,50))
    data = np.array(img)/(255.0)
    X[count,0,:,:] = data
    if names[count] in male:
        Y[count] = 0
    elif names[count] in female:
        Y[count] = 1
    else:
        print('err')
        pdb.set_trace()
    count = count + 1

# first split multiply-occurring patients 70/30
np.random.seed(0)
idx = np.random.permutation(m)
training_idx, test_idx = idx[:int(m*0.7)], idx[int(m*0.7):]
training_ims = []
for idx in training_idx:
    training_ims.extend(ix_people[idx])
training_ims = np.array(training_ims)
test_ims = []
for idx in test_idx:
    test_ims.extend(ix_people[idx])
test_ims = np.array(test_ims)



XTR = X[training_ims,:,:,:]
YTR = Y[training_ims]
XTE = X[test_ims,:,:,:]
YTE = Y[test_ims]
DICT_TR = {}
DICT_TE = {}
DICT_P  = {}
DICT_TR['X'] = XTR
DICT_TE['X'] = XTE
DICT_TR['Y'] = YTR
DICT_TE['Y'] = YTE
DICT_P['training_idx'] = training_idx
DICT_P['test_idx'] = test_idx
DICT_P['training_ims'] = training_ims
DICT_P['test_ims'] = test_ims
name = 'faces'
np.savez_compressed(name + '_train', **DICT_TR)
np.savez_compressed(name + '_test',  **DICT_TE)
np.savez_compressed(name + '_info',  **DICT_P)
pdb.set_trace()
