#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 01:25:54 2021

@author: obaiga
"""

# In[Package]
import numpy as np
# import os 
import pandas as pd
#from collections import Counter
from os.path import join,exists
from os import makedirs
from skimage.io import imread
import glob
# from scipy.io import loadmat
import matplotlib.pyplot as plt
# import copy
import cv2
# import matplotlib.cm as cm
np.set_printoptions(precision = 2)
from shutil import copyfile
import matplotlib.image as image
 # In[Dir]
# =============================================================================
#   Initilization
# =============================================================================

### New database path
dpath = '/Users/obaiga/Jupyter/Python-Research/Africaleopard/'
# dpath = '/Users/obaiga/Research/Snow Leopard/'
###Database name
# new_db = 'non_iso'
# new_db = 'bright40'
new_db = 'snow leopard'

db_dir = join(dpath,new_db)
# mask_dir = join(db_dir,'test_res')
img_dir = join(db_dir,'images')


table = pd.read_csv(join(db_dir,'table.csv'),skipinitialspace=True)
cids = list(np.array((table['#   ChipID']),dtype=int))

img_lis = list(table['Image'])
chip_name = 'cid%d_CHIP(sz%d).png'
                         
cp_dir = join(db_dir,'_hsdb/computed')

kpts_dir = join(db_dir,cp_dir,'feats')

poly_dir = join('/Users/obaiga/Jupyter/Python-Research/ds_160/','Polygon')
# matched_dir = join(db_dir,cp_dir,'matched_kpts')

chipname = 'cid%d_FEAT(hesaff+sift,0_9001)_CHIP(sz750).npz'

res_dir = join(db_dir,'results')
if not exists(res_dir):
    makedirs(res_dir)


# In[Func]
# RCOS TODO: Parametarize interpolation method
INTERPOLATION_TYPES = {
    'nearest': cv2.INTER_NEAREST,
    'linear':  cv2.INTER_LINEAR,
    'area':    cv2.INTER_AREA,
    'cubic':   cv2.INTER_CUBIC,
    'lanczos': cv2.INTER_LANCZOS4
}

cv2_flags = INTERPOLATION_TYPES['lanczos']
cv2_borderMode  = cv2.BORDER_CONSTANT
cv2_warp_kwargs = {'flags': cv2_flags, 'borderMode': cv2_borderMode}

def extract_chip(img_fpath, roi, theta, new_size):
    'Crops chip from image ; Rotates and scales; Converts to grayscale'
    # Read parent image
    #printDBG('[cc2] reading image')
    imgBGR = cv2.imread(img_fpath, cv2.IMREAD_COLOR)
    #printDBG('[cc2] building transform')
    # Build transformation
    (rx, ry, rw, rh) = roi
    (rw_, rh_) = new_size
    Aff = build_transform(rx, ry, rw, rh, rw_, rh_, theta)
    #printDBG('[cc2] rotate and scale')
    # Rotate and scale
    imgBGR = cv2.warpAffine(imgBGR, Aff, (rw_, rh_), **cv2_warp_kwargs)
    #printDBG('[cc2] return extracted')
    return imgBGR

def compute_chip(img_fpath, chip_fpath, roi, theta, new_size, filter_list, force_gray=False):
    '''Extracts Chip; Applies Filters; Saves as png'''
    #printDBG('[cc2] extracting chip')
    chipBGR = extract_chip(img_fpath, roi, theta, new_size)
    #printDBG('[cc2] extracted chip')
    for func in filter_list:
        #printDBG('[cc2] computing filter: %r' % func)
        chipBGR = func(chipBGR)
    cv2.imwrite(chip_fpath, chipBGR)
    return True

import xml.dom.minidom as minidom


def Read_ROI_data(gname,cdir, type_name = 'leopard'):
    non_annotation_flag = False
    if not exists(gname):
        non_annotation_flag = True
    else:
        domTree = minidom.parse(gname)
        root = domTree.documentElement
        for ele in root.getElementsByTagName('size')[0].childNodes:
            if ele.nodeName == 'width':
                width = ele.childNodes[0].nodeValue
            if ele.nodeName == 'height':
                height = ele.childNodes[0].nodeValue
    
        obj_lis = root.getElementsByTagName('name')
        xy = []
        #     for i in range(len(obj_lis)):
        ### only record the bounding box coordinate of the FIRST 'Leopard' object
        if obj_lis[0].childNodes[0].nodeValue == type_name:
            for ele in root.getElementsByTagName('bndbox')[0].childNodes:
                if ele.nodeName != '#text':
                    xy.append(ele.childNodes[0].nodeValue)   ## xy coordinate
    
        ###------- function:[hsgui]-[guitools]-def select_roi():
            
        xm = int(float(xy[0]))
        xM = int(float(xy[2]))
        ym = int(float(xy[1]))
        yM = int(float(xy[3]))
        xywh = list(map(int, map(round, (xm, ym, xM - xm, yM - ym))))
        # xywh = map(int, map(round, (xm, ym, xM - xm, yM - ym)))  ## Python2
            
        
    if non_annotation_flag == True:
        
        print('using full image size\ name:%s'%(gname[len(cdir)+1:-4]))
        ans = join(img_dir,gname[len(cdir)+1:-4]+'.JPG')
        img = image.imread(ans)
        [width,height,channel] = img.shape
        xm = int(1)
        xM = int(height)
        ym = int(1)
        yM = int(width)
        xywh = list(map(int, map(round, (xm, ym, xM - xm, yM - ym))))  ## Python3
        # xywh = map(int, map(round, (xm, ym, xM - xm, yM - ym)))  ## Python2
        print(('Second',xywh))
        
    roi = np.array(xywh, dtype=np.int32)
    
    return roi

def compute_uniform_area_chip_sizes(roi_list, sqrt_area=None):
    'Computes a normalized chip size to rescale to'
    if not (sqrt_area is None or sqrt_area <= 0):
        target_area = sqrt_area ** 2

        def _resz(w, h):
            try:
                ht = np.sqrt(target_area * h / w)
                wt = w * ht / h
                return (int(round(wt)), int(round(ht)))
            except Exception:
                msg = '[cc2.2] Your csv tables have an invalid ROI.'
                print(msg)
                # warnings.warn(msg)
                return (1, 1)
        chipsz_list = [_resz(float(w), float(h)) for (x, y, w, h) in roi_list]
    else:  # no rescaling
        chipsz_list = [(int(w), int(h)) for (x, y, w, h) in roi_list]
    return chipsz_list

def build_transform(x, y, w, h, w_, h_, theta, homogenous=False):
    sx = (w_ / w)  # ** 2
    sy = (h_ / h)  # ** 2
    cos_ = np.cos(-theta)
    sin_ = np.sin(-theta)
    tx = -(x + (w / 2))
    ty = -(y + (h / 2))
    T1 = np.array([[1, 0, tx],
                   [0, 1, ty],
                   [0, 0, 1]], np.float64)

    S = np.array([[sx, 0,  0],
                  [0, sy,  0],
                  [0,  0,  1]], np.float64)

    R = np.array([[cos_, -sin_, 0],
                  [sin_,  cos_, 0],
                  [   0,     0, 1]], np.float64)

    T2 = np.array([[1, 0, (w_ / 2)],
                   [0, 1, (h_ / 2)],
                   [0, 0, 1]], np.float64)
    M = T2.dot(R.dot(S.dot(T1)))
    #.dot(R)#.dot(S).dot(T2)

    if homogenous:
        transform = M
    else:
        transform = M[0:2, :] / M[2, 2]
        
    return transform
# In[Chip Mask]
'''
Generate a modified mask with chip size (Hotspotter resize a leopard ROI)
'''

mask_dir  = db_dir

sqrt_area = 750
theta = 0
filter_list = []
name = 'cid%d_CHIP(sz%d).png'
# poly_dir = join(db_dir,'Polygon')
seg_class_name = 'leopard'
mask_chip_dir = join(db_dir,'mask_chip')
if not exists (mask_chip_dir):
    makedirs(mask_chip_dir)
    
roi_lis = table['roi[tl_x  tl_y  w  h]']    

for i,icid in enumerate(cids):
    iimg = img_lis[i]
    iimg = iimg[:-4]+'.JPG'
    mask_fpath = join(mask_dir,'mask',iimg)
    if not exists (mask_fpath):
        print('no mask: cid:%d \n name:%s'%(icid,iimg))
        pass
    else:

        # imat = join(poly_dir,iimg[:-4]+'.xml')
        # roi = Read_ROI_data(imat,poly_dir,seg_class_name)
        
        roi_str = roi_lis[i].strip('[').strip(']')
        roi = [int(round(float(_))) for _ in roi_str.split()]
            
        chipsz = compute_uniform_area_chip_sizes([roi], sqrt_area)
        ans = join(mask_chip_dir,name%(icid,sqrt_area))
        compute_chip(mask_fpath,ans, roi, theta, chipsz[0], filter_list, force_gray=False)


# In[Retain kpts in template]
'''
# =============================================================================
#     Replace chip only keep keypoints in the leopard template
# =============================================================================
'''
mask_chip_dir = join(db_dir,'mask_chip')
name = 'cid%d_CHIP(sz%d).png'
sqrt_area = 750

lis = cids
check = np.zeros([len(lis),2])
feat_ori_dir = join(cp_dir,'feats_ori')
feat_new_dir = join(cp_dir,'feats_fg')
if not exists(feat_new_dir):
    makedirs(feat_new_dir)

for i,icid in enumerate(lis):
    
    # ======================================================
    #     load kpts & segmentation from query images 
    # ======================================================
    i_path = join(feat_ori_dir,(chipname%icid))
    chip_fg_path = join(mask_chip_dir,name%(icid,sqrt_area))
    if exists(chip_fg_path):
        fg_query = np.array(imread(chip_fg_path)/255,dtype=np.int8)
        try:
            npz = np.load(i_path,mmap_mode=None)
            kpts = npz['arr_0']
            desc = npz['arr_1']
            # print(npz.files)
            check[i,0] = len(kpts)
            
            arr_0 = []
            arr_1 = []
            for ii in range(len(kpts)):
                xy_q = np.array(np.ceil(kpts[ii][:2]),dtype=np.int32)
                flag_fg_q = fg_query[xy_q[1],xy_q[0]]  
                if flag_fg_q[0] == 1:
                    arr_0.append(kpts[ii,:])
                    arr_1.append(desc[ii,:])
            
            check[i,1] = len(arr_0)
            if len(arr_0) < 1:
                print('no keypoints: cid %d\n name:%s'%(icid,img_lis[i]))
            # ======================================================
            #     save
            # ======================================================                    
            np.savez((join(feat_new_dir,(chipname%icid))),arr_0=arr_0,arr_1=arr_1)
        except IOError:
                print('Wrong query idx for reading kpts:%r'%icid)
    else:
        print('no maskchip exit: cid:%d\n name:%s'%(icid,img_lis[i])) 
        copyfile(i_path, join(feat_new_dir,chipname%icid))
    
df = pd.DataFrame(check)
sentence = join(res_dir,'Kpts_fg_HS.xlsx')
df.to_excel(sentence, index=False, header=False)   


# In[create customed mask-image]

# cid = [1552,567,568,1250,1251,1252]

sqrt_area = 750
cp_dir = join(db_dir,'_hsdb/computed')
chip_dir = join(cp_dir,'chips')
mask_chip_dir = join(db_dir,'mask_chip')
name = 'cid%d_CHIP(sz%d).png'

save_dir = join(db_dir,'PlotResult','customed mask image')
if not exists(save_dir):
    makedirs(save_dir)
    
ans = glob.glob(join(chip_dir,'*.png'))
ans = np.sort(ans)
# for icid in cid:
for ians in ans:
    # iname = name %(icid,sqrt_area)
    iname = ians[len(chip_dir)+1:]
    img = imread(join(chip_dir,iname))
    mask_dir = join(mask_chip_dir,iname)
    if exists (mask_dir):
        mask = imread(mask_dir,as_gray = True)
        mask_3d = np.zeros((mask.shape[0], mask.shape[1],3))
        mask_3d[:,:,0] = mask
        mask_3d[:,:,1] = mask 
        mask_3d[:,:,2] = mask
        obj_image = img * mask_3d
        
        save_name = join(save_dir,iname)
        plt.imsave(save_name, obj_image.astype(np.uint8))
    else:
        copyfile(ians, join(save_dir,iname))
        print(iname)

    
    # _,ax = plt.subplots(1,1)
    # ax.imshow(obj_image.astype(np.uint8))
    # ax.imshow(mask_req,cmap=plt.cm.gray)