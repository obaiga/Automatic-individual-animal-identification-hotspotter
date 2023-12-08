#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 12:59:02 2023

@author: obaiga
"""

import pandas as pd
from os.path import join


### New database path
dpath = '/Users/obaiga/Jupyter/Python-Research/Africaleopard/'

###Database name
new_db = 'bright20'
new_db = 'snowleop'

db_dir = join(dpath,new_db)

table_dir = join(db_dir,'flat_table.csv')
table = pd.read_csv(table_dir,header=2)
table_dir = join(db_dir,'table.csv')
table.to_csv(table_dir,index=0)

cache_dir = join(db_dir,'_hsdb')
path = join(cache_dir,'name_table.csv')
table_ID = pd.read_csv(path,header=2)
path = join(db_dir,'name_table.csv')
table_ID.to_csv(path,index=0)