# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 12:14:40 2021
@author: obaiga

------------------
Create an Africa leopard image dataset
------------------
"""
# In[packages]
from __future__ import division, print_function
from os.path import join, expanduser,split, exists
import numpy as np
import xml.dom.minidom as minidom
import sys 
import matplotlib.image as image
import shutil
from os import chdir

# Set the path to your desired directory
main_directory = '/Users/obaiga/github/Automatic-individual-animal-identification-hotspotter/hotspotter/'

# Change the current working directory
chdir(main_directory)

### from Hotspotter
from hscom import helpers
from hscom import argparse2
from hscom import params
from hscom import fileio as io
from hscom import __common__
from hotspotter import HotSpotterAPI
(print, print_, print_on, print_off,
 rrr, profile) = __common__.init(__name__, '[helpers]')

HOME = expanduser('~')
GLOBAL_CACHE_DIR = join(HOME, '.hotspotter/global_cache')
helpers.ensuredir(GLOBAL_CACHE_DIR)

from collections import Counter 
import pandas as pd

# In[functions]
# =============================================================================
#   Function 
# =============================================================================

def parse_arguments(defaultdb, usedbcache):
    args = argparse2.parse_arguments(defaultdb=defaultdb)
    # Parse arguments
    args = argparse2.fix_args_with_cache(args)
    if usedbcache:
        if args.vdd:
            helpers.vd(args.dbdir)
            args.vdd = False
        if helpers.inIPython() or '--cmd' in sys.argv:
            args.nosteal = True
    params.args = args
    # Preload process args
    if args.delete_global:
        io.delete_global_cache()
    return args

def preload_args_process(args):
    from hscom import helpers
    import sys
    # Process relevant args
    cids = args.query
    if args.vdd:
        helpers.vd(args.dbdir)
        args.vdd = False
    load_all = args.autoquery or len(cids) > 0
    if helpers.inIPython() or '--cmd' in sys.argv:
        args.nosteal = True
    return load_all, cids

def open_database(db_dir=None):
    # File -> Open Database
    try:
        # Use the same args in a new (opened) database
        args = params.args
        #args = back.params.args

        # Try and load db
        if args is not None:
            args.dbdir = db_dir
        hs = HotSpotterAPI.HotSpotter(args=args, db_dir=db_dir)
        hs.load(load_all=False)

    except Exception as ex:
        import traceback
        import sys
        print(traceback.format_exc())
        print('aborting open database')
        print(ex)
        if '--strict' in sys.argv:
            raise
        raise
    print('')
    return hs

def Read_ROI_data(gx,plygon_dpath,type_name = 'leopard',Flag_add_chip_software=False):
    non_annotation_flag = False
    gname = hs.tables.gx2_gname[gx]
    
    if Flag_add_chip_software == True:
        cname = gname[:-3] + 'xml'
        cdir = join(plygon_dpath,cname)
        if not exists(cdir):
            non_annotation_flag = True
        else:
            domTree = minidom.parse(cdir)
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
            
        
    if Flag_add_chip_software == False or non_annotation_flag == True:
        print('using full image size, idx:%d \ name:%s'%(gx,gname))
        img_dir = join(img_dpath,gname)
        img = image.imread(img_dir)
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

# In[create dataset]
# =============================================================================
#     Initialization -  dataset (create a new dataset)
# =============================================================================
### New database path
dpath = '/Users/obaiga/Jupyter/Python-Research/Africaleopard'
###Database name
new_db = 'snow leopard'

db_dir = join(dpath, new_db)
### Full path: dapth + new_db

### Whether add a new database
Flag_new_db = True

if Flag_new_db & 0:
    if exists(db_dir):
        shutil.rmtree(db_dir)

#------ function:[hsgui]-[guitools]- def select_directory
caption='Select Directory'
print('Selected Directory: %r' % dpath)
io.global_cache_write('select_directory', split(dpath)[0])
print('[*back] valid new_db_dir = %r' % db_dir)
io.global_cache_write('db_dir', db_dir)
helpers.ensurepath(db_dir)

defaultdb = None
preload = False

args = parse_arguments(defaultdb, defaultdb == 'cache')
# --- Build HotSpotter API ---
hs = open_database(db_dir)


# In[load images]
# =============================================================================
#     Initialization -  images  (load your dataset images)
# =============================================================================   
#-------function: [hsgui]-[guiback]-def import_images_from_dir(back):    
#--------------------------------------------
'''
custom by yourself 
here is an example
'''
#%%
name = 'raw_data.csv'
table_dir = join(dpath,new_db,name)
img_dpath = '/Users/obaiga/Jupyter/Python-Research/Africaleopard/snow leopard/images_db/'
table = pd.read_csv(table_dir,skipinitialspace=True)

# name = 'table.csv'
# table_dir = join(dpath,name)
# table = pd.read_csv(table_dir,skipinitialspace=True)
# img_dpath = join('/Users/obaiga/Jupyter/Python-Research/Africaleopard',new_db,'img_db')

#### print the table column name
#### print(table.columns.tolist())

Lis_Chip2ID = np.array(table['Name'])
Lis_Img = np.array(table['Image'])
Lis_ROI = np.array(table['roi[tl_x  tl_y  w  h]'])


Lis_img_save = []
Lis_chip2ID_save = []
Lis_ROI_save = []
#%%
###### for non-isolated images
# IDs_info = np.array(Counter(Lis_Chip2ID).most_common())
# idx_save = np.where(IDs_info[:,1].astype(np.int32) > 1)[0]
##### only search for classes with required class size

# for iidx in idx_save:
#     ans = np.where(Lis_Chip2ID == IDs_info[iidx,0])[0]
#     Lis_img_save = np.concatenate((Lis_img_save,Lis_Img[ans]))
#     Lis_chip2ID_save = np.concatenate((Lis_chip2ID_save,Lis_Chip2ID[ans]))
#     Lis_ROI_save = np.concatenate((Lis_ROI_save,Lis_ROI[ans]))
##########----------------------------
import copy

Lis_img_save = copy.deepcopy(Lis_Img)
chip_ID_lis = copy.deepcopy(Lis_Chip2ID)
Lis_ROI_save = copy.deepcopy(Lis_ROI)


fpath_list = []

for iidx,iimg in enumerate(Lis_img_save):

    fpath_list.append(join(img_dpath,iimg))
    # chip_ID_lis.append(Lis_chip2ID_save[iidx])


#%%
Flag_add_img = True

if Flag_add_img & 1:
    print('[*back] selected %r' % img_dpath)
    # fpath_list = helpers.list_images(img_dpath, fullpath=True)
    hs.add_images(fpath_list)
#        hs.add_templates(img_dpath)
    print('')
    
'''     
#------function def import_images_from_file(back):
# File -> Import Images From File
#hs.add_images(fpath_list)
'''
    

# In[load chips]
# =============================================================================
#     Initialization -  chips
# =============================================================================
'''
custom by yourself
'''
Flag_add_chip = True

if Flag_add_chip & 1:
    #-------function: [hsgui]-[guiback] - def add_chip(back):
    
    gx_lis = hs.get_valid_gxs()
    gx_lis_valid = gx_lis[len(gx_lis)-len(fpath_list):]

    for gx in gx_lis_valid:
        
        roi_str = Lis_ROI_save[gx].strip('[').strip(']')
        roi = [int(round(float(_))) for _ in roi_str.split()]

        cx = hs.add_chip(gx, roi)  # NOQA
#    #    back.select_gx(gx)
        print('')
# In[load IDs]
# =============================================================================
#     Initialization -   new leopard ID name
# =========================================================================
'''
custom by yourself
'''
Flag_chip_ID  = True
note = ''

if Flag_chip_ID & 1:
    #----function: [hsgui]-[guiback]- def change_chip_property(back, cid, key, val):
    key = 'name'
    # Table Edit -> Change Chip Property
    cx_lis = hs.get_valid_cxs()
    cx_lis_valid = cx_lis[len(cx_lis)-len(fpath_list):]
    
    for cx,val in zip(cx_lis_valid,chip_ID_lis):
    
        key, val = map(str, (key, (str(val))))
        
        # key, val = map(str, (key, (note+'_'+str(val))))
        
        print('[*back] change_chip_property(%r, %r, %r)' % (cx, key, val))
        if key in ['name', 'matching_name']:
            hs.change_name(cx, val)
        else:
            hs.change_property(cx, key, val)
        print('') 
        
# In[Save database]
# =============================================================================
#     Update database
# =============================================================================        
hs.save_database()

# In[Create chip]
# =============================================================================
#         Load Chip & Feature 
# =============================================================================
cx_list = None     


cx_list = hs.get_valid_cxs() if cx_list is None else cx_list
    # Explain function-[hs.get_valid_cxs()]
    # valid_cxs = np.where(hs.tables.cx2_cid > 0)[0] 
if not np.iterable(cx_list):
    valid_cxs = [cx_list]
cx_list = np.array(cx_list)  # HACK
   
hs.load_chips(cx_list=cx_list)
# hs.load_features(cx_list=cx_list)

        
# In[Write flat table]
# =============================================================================
#    Create flat table (Hotspotter (Original version))
# =============================================================================
from hotspotter import load_data2 as ld2
def write_flat_table(hs):
    dbdir = hs.dirs.db_dir
    # Make flat table
    valid_cx = hs.get_valid_cxs()
    flat_table  = make_flat_table(hs, valid_cx)
    flat_table_fpath  = join(dbdir, 'flat_table.csv')
    # Write flat table
    print('[ld2] Writing flat table')
    helpers.write_to(flat_table_fpath, flat_table)
    
def make_flat_table(hs, cx_list):
    # Valid chip tables
    if len(cx_list) == 0:
        return ''
    cx2_cid   = hs.tables.cx2_cid[cx_list]
    # Use the indexes as ids (FIXME: Just go back to g/n-ids)
    cx2_gname = hs.cx2_gname(cx_list)
    cx2_name  = hs.cx2_name(cx_list)
    try:
        cx2_roi   = hs.tables.cx2_roi[cx_list]
    except IndexError as ex:
        print(ex)
        cx2_roi = np.array([])
    cx2_theta = hs.tables.cx2_theta[cx_list]
    prop_dict = {propkey: [cx2_propval[cx] for cx in iter(cx_list)]
                 for (propkey, cx2_propval) in hs.tables.prop_dict.items()}
    # Turn the chip indexes into a DOCUMENTED csv table
    header = '# flat table'
    column_labels = ['ChipID', 'Image', 'Name', 'roi[tl_x  tl_y  w  h]', 'theta']
    column_list   = [cx2_cid, cx2_gname, cx2_name, cx2_roi, cx2_theta]
    column_type   = [int, int, int, list, float]
    if not prop_dict is None:
        for key, val in prop_dict.items():
            column_labels.append(key)
            column_list.append(val)
            column_type.append(str)

    chip_table = ld2.make_csv_table(column_labels, column_list, header, column_type)
    return chip_table

write_flat_table(hs)

#%%
# import pandas as pd
table_dir = join(db_dir,'flat_table.csv')
table = pd.read_csv(table_dir,header=2)
table_dir = join(db_dir,'table.csv')
table.to_csv(table_dir,index=0)

cache_dir = join(db_dir,'_hsdb')
path = join(cache_dir,'name_table.csv')
table_ID = pd.read_csv(path,header=2)
path = join(db_dir,'name_table.csv')
table_ID.to_csv(path,index=0)


#%%
# =============================================================================
#     Main Query Part (cannot operate in python3.7)
# =============================================================================
#Detailed function: [hsgui]-[guiback]-function: query()
if 0:
    cx_list = hs.get_valid_cxs()
    for cx in cx_list:
        cid = hs.cx2_cid(cx)
        print('[back] cx = %r' % cx)
        print('[back] query(cid=%r)' % (cid))
        
        try:
            res = hs.query(cx)
        except Exception as ex:
            # TODO Catch actually exceptions here
            print('[back] ex = %r' % ex)
            raise