# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 16:26:53 2021
@author: obaiga

------------------
Function: Save number of matched keypoints for image pairs 
        & Save image similarity score for image pairs

Query function: Hotspotter 1vsmany
------------------
"""
# In[package]
import numpy as np
import os 
import pandas as pd
#from collections import Counter
from os.path import join,exists
from os import makedirs

np.set_printoptions(precision = 2)

# In[init]
# =============================================================================
#  Initialization (User needs to modify the below contens )
# =============================================================================

### New database path
dpath = '/Users/obaiga/Jupyter/Python-Research/Africaleopard/'
###Database name
new_db = 'non_iso'
new_db = 'bright60'
new_db = 'snow leopard'

# flag = 'vsone_'   ## flag = 'vsone' or 'vsmany'
flag = 'vsmany'   ## flag = 'vsone' or 'vsmany'
fg_flag = 'fg'   #### only containing animal body without background
# fg_flag = ''

# =============================================================================
#   Initilization
# =============================================================================
db_dir = join(dpath,new_db)
path = join(db_dir,'table.csv')
table = pd.read_csv(path,skipinitialspace=True)

cids = list(np.array((table['#   ChipID']),dtype=int))
                          
if flag=='vsone':
    pre_file = 'res_yl}@``k]tbl~_%o`_qcid='
    pre_save = 'vsone_'
elif flag =='vsmany':
    pre_file = 'res_w1=@2,y0es`5;[=]_qcid='
    pre_save = 'vsmany_'
    
result_dir = join(db_dir,'_hsdb','computed','query_results')
kpts_dir = join(db_dir,'_hsdb','computed','feats')
    
chipname = 'cid%d_FEAT(hesaff+sift,0_9001)_CHIP(sz750).npz'

res_dir = join(db_dir,'results')
if not exists(res_dir):
    makedirs(res_dir)
    
# In[Save Kpts]
# =============================================================================
#   Output table about keypoint number for images 
# =============================================================================
num_kpts = []
for icid in cids:
    i_chip = ''.join(chipname%icid)
    i_path = os.path.join(kpts_dir,i_chip)
    try:
        npz = np.load(i_path,mmap_mode=None)
        kpts = npz['arr_0']
        desc = npz['arr_1']
        num_kpts.append(len(kpts))
    except IOError:
        print('Cannot read file, Idx:%r'%icid)

data = np.array([cids,num_kpts],dtype=np.int32)
data = data.T
df = pd.DataFrame(data)

sentence = join(res_dir,'Kpts_%sHS.xlsx'%fg_flag)
df.to_excel(sentence, index=False, header=False)

# In[Save Query Result]
# =============================================================================
#   Save number of mathced keypoints for each image 
#   & Image similarity score matrix
# =============================================================================

num = len(cids)
score_array = np.zeros([num,num])
matched_kpts_array = np.zeros([num,num])
for i,icid in enumerate(cids):
    file = pre_file+str(icid)+'.npz'
    try: 
        data = np.load(join(result_dir,file),encoding="latin1",allow_pickle=True)
        #__slots__ = ['true_uid', 'qcx', 'query_uid', 'uid', 'title', 'nn_time',
             #'weight_time', 'filt_time', 'build_time', 'verify_time',
             #'cx2_fm', 'cx2_fs', 'cx2_fk', 'cx2_score']
        # print(lis(data.keys()))
        score = data['cx2_score']
        matched_kpts = data['cx2_fm']
        score_array[i,:] = score
        for j,i_matched in enumerate(matched_kpts):
            matched_kpts_array[i,j] = len(i_matched)
    except:
        print('Cannot read file:'+file)

df = pd.DataFrame(score_array)
sentence = join(res_dir,('ImgScore_%s%s.xlsx')%(pre_save[2:],fg_flag))
df.to_excel(sentence, index=False, header=False)

df = pd.DataFrame(matched_kpts_array)
sentence = join(res_dir,('MatchKpts_%s%s.xlsx')%(pre_save[2:],fg_flag))
df.to_excel(sentence, index=False, header=False)


# In[modify similarity score]
data_mof = '_diag'
import copy 
[weight,height] = score_array.shape

score_mod = copy.copy(score_array)

for irow in range(weight):
    for jcolumn in range(weight):
        if irow == jcolumn:
            value = np.sum(score_array[irow,:])
            score_mod[irow,jcolumn] = value
            if value == 0:
                score_mod[irow,jcolumn] = 1
        
                  
df = pd.DataFrame(score_mod)
sentence = join(res_dir,('ImgScore_%s%s%s.xlsx')%(pre_save[2:],fg_flag,data_mof))
df.to_excel(sentence, index=False, header=False)