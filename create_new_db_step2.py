# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 12:14:40 2021
@author: obaiga

------------------
Create own database
Replace Hotspotter function: create new database; add images; add chips; modify ID name
------------------

Please before running the program, modify Initilization part 
'dpath', 'new_db', 'image_dpath'
Now, default 'annotation_dpath' is 'image_dpath'
default chip size is original image size.

If you want to extract ROI, set 'Flag_add_chip_software' = True and 'chip_dpath'
modify function 'Read_ROI_data' based on your ROI formate.

"""
#%%
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


#%%
# =============================================================================
#  Initialization (User needs to modify the below contens )
# =============================================================================
### New database path

# dpath = 'C:\\Users\\95316\\code1\\Snow leopard'
dpath = '/Users/obaiga/Jupyter/Python-Research/'
dpath = '/Users/obaiga/Research/Snow Leopard/'
dpath = '/Users/obaiga/github/hotspotter/'

###Database name
new_db = 'ds_160'
# new_db = 'Test2'
db_dir = join(dpath, new_db)
### Full path: dapth + new_db

### Whether add a new database
Flag_new_db = False

### Whether add new images 
Flag_add_img = False
# img_dpath = 'C:\\Users\\95316\\code1\\Snow leopard\\RepresentativeTests_right_Cat1'
img_dpath = join(db_dir,'images-db')

### whether add new chips 
Flag_add_chip = False
Flag_add_chip_software = True     ## True: chip size created by Rectle;
                                 ## False: full orginal image size 
chip_dpath = join(db_dir,'Polygon')
#chip_dpath = 'C:\Users\95316\code1\Snow leopard\RepresentativeTests_right_diff_cats\annotation'
### Chip (read xml files)
seg_class_name = 'leopard'

### whether add new ID name (only works on already having chips)
Flag_chip_ID = True
# chip_ID_lis = ['Cat'+str(i) for i in range(9)]

#%%
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

def Read_ROI_data(gx, type_name = 'leopard',Flag_add_chip_software=False):
    non_annotation_flag = False
    gname = hs.tables.gx2_gname[gx]
    
    if Flag_add_chip_software == True:
        cname = gname[:-3] + 'xml'
        cdir = join(chip_dpath,cname)
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

#%%
# =============================================================================
#     Initialization -  dataset
# =============================================================================


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


#%%
# =============================================================================
#     Initialization -  images  
# =============================================================================   
#-------function: [hsgui]-[guiback]-def import_images_from_dir(back):
    
#--------------------------------------------
from collections import Counter 
import pandas as pd
# side = 'Right'
# label = 'R'
side = 'Left'
label = 'L'

count_db = 1
table_dir = join(db_dir,'table-%s-1.csv'%side)
table = pd.read_csv(table_dir,skipinitialspace=True)

ID_name_lis = np.array(table['Name'])
img_name_lis = np.array(table['Image'])



fpath_list = []
chip_ID_lis = []

ID_count = Counter(ID_name_lis).most_common()
Count_lis = []
for i_IDname,i_Count in ID_count:
    Count_lis.append(i_Count)
    if i_Count >= count_db:
        imgcx = np.where(ID_name_lis==i_IDname)
        imgs = img_name_lis[imgcx]
        for i_img in imgs:
            fpath_list.append(join(img_dpath,i_img))
            chip_ID_lis.append(i_IDname)
print(Counter(Count_lis).most_common())


#%%
if 0:
    import glob
    img_dpath = join(db_dir,'images-db')
    fpath_list = glob.glob(join(img_dpath,'*.JPG'))

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
    

#%%
# =============================================================================
#     Initialization -  chips
# =============================================================================
# Flag_add_chip_table = True
# Flag_add_chip_software = False
Flag_add_chip_table = False

if Flag_add_chip & 1:
    #-------function: [hsgui]-[guiback] - def add_chip(back):
    
    gx_lis = hs.get_valid_gxs()
    gx_lis_valid = gx_lis[len(gx_lis)-len(fpath_list):]
    
    if Flag_add_chip_table & 1:
        table_dir = join(db_dir,'1674_Leopard_list.csv')
        table = pd.read_csv(table_dir,skipinitialspace=True)
        img_name_all = np.array(table['FileName'])
        Xmin_lis = np.array(table['Xmin'])
        Ymin_lis = np.array(table['Ymin'])
        Xmax_lis = np.array(table['Xmax'])
        Ymax_lis = np.array(table['Ymax'])
        
    for gx in gx_lis_valid:
        
        if Flag_add_chip_software & 1:
            roi = Read_ROI_data(gx,seg_class_name,Flag_add_chip_software)
        elif Flag_add_chip_table & 1:
            gname = hs.tables.gx2_gname[gx]
            idx = np.where(img_name_all==gname)[0]
            if len(idx)>0:
                xm = int(Xmin_lis[idx[0]])
                xM = int(Xmax_lis[idx[0]])
                ym = int(Ymin_lis[idx[0]])
                yM = int(Ymax_lis[idx[0]])
                # xywh = map(int, map(round, (xm, ym, xM - xm, yM - ym)))
                xywh = [xm, ym, xM - xm, yM - ym]
                roi = np.array(xywh, dtype=np.int32)
            else:
                print('wrong image: %d'%gx)
        else:
            roi = Read_ROI_data(gx,Flag_add_chip_software)
        
        roi = np.array(roi,dtype=np.int32)
        
        cx = hs.add_chip(gx, roi)  # NOQA
#    #    back.select_gx(gx)
        print('')
#%%
if Flag_chip_ID & 1:
    #----function: [hsgui]-[guiback]- def change_chip_property(back, cid, key, val):
    key = 'name'
    # Table Edit -> Change Chip Property
    cx_lis = hs.get_valid_cxs()
    cx_lis_valid = cx_lis[len(cx_lis)-len(fpath_list):]
    for cx,val in zip(cx_lis_valid,chip_ID_lis):
        
        if val[:5] == 'Leopa':
            
            note = 'Leopard_African_'
            print(note)
            key, val = map(str, (key, (val[len(note):]+'_'+side)))
        else:
            note = ''
            key, val = map(str, (key, (val[len(note):]+'_'+side)))
        # key, val = map(str, (key, (val[len(note):])))
        print('[*back] change_chip_property(%r, %r, %r)' % (cx, key, val))
        if key in ['name', 'matching_name']:
            hs.change_name(cx, val)
        else:
            hs.change_property(cx, key, val)
        print('') 
# In[]
# # =============================================================================
# #     Initialization -  new leopard ID name
# # =============================================================================
# if Flag_chip_ID == True & 0:
#     #----function: [hsgui]-[guiback]- def change_chip_property(back, cid, key, val):
#     key = 'name'
#     # Table Edit -> Change Chip Property
#     cx_lis = hs.get_valid_cxs()
#     for cx,val in zip(cx_lis,chip_ID_lis):
#         key, val = map(str, (key, val))
#         print('[*back] change_chip_property(%r, %r, %r)' % (cx, key, val))
#         if key in ['name', 'matching_name']:
#             hs.change_name(cx, val)
#         else:
#             hs.change_property(cx, key, val)
#         print('')        
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

#%%
# =============================================================================
#     Main Query Part
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