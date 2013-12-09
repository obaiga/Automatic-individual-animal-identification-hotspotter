from __future__ import division, print_function
import __builtin__
import sys
from itertools import izip
import fileio as io
from os.path import realpath
import PyQt4
import draw_func2 as df2
import vizualizations as viz
from PyQt4 import Qt, QtCore, QtGui
from PyQt4.Qt import pyqtSlot, pyqtSignal
import guifront
import guitools

# Dynamic module reloading
def reload_module():
    import imp, sys
    print('[guiback] reloading '+__name__)
    imp.reload(sys.modules[__name__])
rrr = reload_module

class MainWindowBackend(QtCore.QObject):
    ''' Class used by the backend to send and recieve signals to and from the
    frontend'''
    populateSignal = pyqtSignal(str, list, list, list, list)
    def __init__(self, hs=None):
        super(MainWindowBackend, self).__init__()
        print('[back] creating backend')
        self.hs = hs
        self.win = guifront.MainWindowFrontend(backend=self)
        self.selection = None
        df2.register_matplotlib_widget(self.win.plotWidget)
        # connect signals
        self.populateSignal.connect(self.win.populate_tbl)
        if hs is not None:
            self.connect_api(hs)
            
    def connect_api(self, hs):
        print('[win] connecting api')
        self.hs = hs
        self.populate_image_table()
        self.populate_chip_table()
        #self.database_loaded.emit()

    def populate_image_table(self):
        print('[win] populate_image_table()')
        #col_headers  = ['Image ID', 'Image Name', 'Chip IDs', 'Chip Names']
        #col_editable = [ False    ,  False      ,  False    ,  False      ]
        col_headers   = ['Image Index', 'Image Name', 'Num Chips']
        col_editable  = [False, False, False]
        # Populate table with valid image indexes
        gx2_gname = self.hs.tables.gx2_gname
        gx2_cxs   = self.hs.get_gx2_cxs()
        row_list  = range(len(gx2_gname))
        row2_datatup = [(gx, gname, len(gx2_cxs[gx])) for gx, gname in enumerate(gx2_gname)]
        self.populateSignal.emit('image', col_headers, col_editable, row_list, row2_datatup)

    def populate_chip_table(self):
        print('[win] populate_chip_table()')
        col_headers  = ['Chip ID', 'Name', 'Image']
        col_editable = [ False  ,    True,   False]
        # Add User Properties to headers
        prop_dict = self.hs.tables.prop_dict
        prop_keys = prop_dict.keys()
        col_headers  += prop_keys
        col_editable += [True]*len(prop_keys)
        # Populate table with valid image indexes
        cx2_cid  = self.hs.tables.cx2_cid
        cx2_nx   = self.hs.tables.cx2_nx
        cx2_gx   = self.hs.tables.cx2_gx
        nx2_name = self.hs.tables.nx2_name
        gx2_gname = self.hs.tables.gx2_gname
        cx_list = [cx for cx, cid in enumerate(cx2_cid) if cid > 0]
        # Build lists
        cid_list   = [cx2_cid[cx] for cx in iter(cx_list)]
        name_list  = [nx2_name[cx2_nx[cx]]  for cx in iter(cx_list)]
        image_list = [gx2_gname[cx2_gx[cx]] for cx in iter(cx_list)]
        prop_lists = [prop_dict[key] for key in prop_keys]
        datatup_iter = izip(*[cid_list, name_list, image_list] + prop_lists)
        row2_datatup = [tup for tup in datatup_iter]
        row_list  = range(len(row2_datatup))
        self.populateSignal.emit('chip', col_headers, col_editable, row_list, row2_datatup)

    @pyqtSlot(name='open_database')
    def open_database(self):
        print('[back] open_database')
        import HotSpotter
        try:
            args = self.hs.args # Take previous args
            # Ask user for db
            db_dir = select_directory('Select (or create) a database directory.')
            print('[main] user selects database: '+db_dir)
            # Try and load db
            hs = HotSpotter.HotSpotter(args=args, db_dir=db_dir)
            hs.load()
            # Write to cache and connect if successful
            io.global_cache_write('db_dir', db_dir)
            self.connect_api(hs)
        except Exception as ex:
            print('aborting open database')
            print(ex)

    @pyqtSlot(name='save_database')
    def save_database(self):
        print('[back] save_database')
        raise NotImplemented('');

    @pyqtSlot(name='import_images')
    def import_images(self):
        print('[back] import_images')
        raise NotImplemented('')

    @pyqtSlot(name='quit')
    def quit(self):
        print('[back] quit()')
        QtGui.qApp.quit()

    def add_images_from_dir(self):
        print('[back] add_images_from_dir')
        img_dpath = select_directory('Select directory with images in it')
        raise NotImplemented('')
    
    def show(self):
        self.win.show()

    @pyqtSlot(str, name='backend_print')
    def backend_print(self, msg):
        print(msg)

    @pyqtSlot(str, name='clear_selection')
    def clear_selection(self):
        print('[back] clear_selection()')
        self.selection = None
        viz.show_splash(self.hs)

    @pyqtSlot(int, name='select_gx')
    def select_gx(self, gx):
        print('[back] select_gx(%r)' % gx)
        self.selection = ('gx', gx)
        viz.show_image(self.hs, gx)

    @pyqtSlot(int, name='select_cid')
    def select_cid(self, cid):
        print('[back] select_cid(%r)' % cid)
        cx = self.hs.cid2_cx(cid)
        viz.show_chip(self.hs, cx)