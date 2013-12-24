from __future__ import division, print_function
from os.path import split
import sys
#import warnings
import numpy as np
import PyQt4
from PyQt4 import Qt, QtCore, QtGui
import fileio as io
import helpers
import draw_func2 as df2

IS_INIT = False
DISABLE_NODRAW = False
DEBUG = False

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s


# Dynamic module reloading
def rrr():
    import imp
    print('[*guitools] reloading ' + __name__)
    imp.reload(sys.modules[__name__])


def configure_matplotlib():
    import multiprocessing
    import matplotlib
    mplbackend = matplotlib.get_backend()
    if multiprocessing.current_process().name == 'MainProcess':
        print('[*guitools] current mplbackend is: %r' % mplbackend)
        print('[*guitools] matplotlib.use(Qt4Agg)')
    else:
        return
    matplotlib.rcParams['toolbar'] = 'toolbar2'
    matplotlib.rc('text', usetex=False)
    #matplotlib.rcParams['text'].usetex = False
    if mplbackend != 'Qt4Agg':
        matplotlib.use('Qt4Agg', warn=True, force=True)
        mplbackend = matplotlib.get_backend()
        if multiprocessing.current_process().name == 'MainProcess':
            print('[*guitools] current mplbackend is: %r' % mplbackend)
        #matplotlib.rcParams['toolbar'] = 'None'
        #matplotlib.rcParams['interactive'] = True


#---------------
# SLOT DECORATORS
def dbgslot_(*types):  # This is called at wrap time to get args
    'wrapper around pyqtslot decorator'

    # Wrap with debug statments
    def pyqtSlotWrapper(func):
        func_name = func.func_name
        print('[@guitools] Wrapping %r with dbgslot_' % func.func_name)

        @Qt.pyqtSlot(*types, name=func.func_name)
        def slot_wrapper(self, *args, **kwargs):
            argstr_list = map(str, args)
            kwastr_list = ['%s=%s' % item for item in kwargs.iteritems()]
            argstr = ', '.join(argstr_list + kwastr_list)
            print('[**dbgslot_] %s(%s)' % (func_name, argstr))
            #with helpers.Indenter():
            result = func(self, *args, **kwargs)
            print('[**dbgslot_] Finished %s(%s)' % (func_name, argstr))
            return result

        slot_wrapper.func_name = func_name
        return slot_wrapper
    return pyqtSlotWrapper


def infoslot_(*types):  # This is called at wrap time to get args
    'wrapper around pyqtslot decorator'

    # Wrap with debug statments
    def pyqtSlotWrapper(func):
        func_name = func.func_name
        #printDBG('[@guitools] Wrapping %r with infoslot_' % func.func_name)

        @Qt.pyqtSlot(*types, name=func.func_name)
        def slot_wrapper(self, *args, **kwargs):
            #printDBG('[**infoslot_] %s()' % (func_name))
            #with helpers.Indenter():
            result = func(self, *args, **kwargs)
            #printDBG('[**infoslot_] Finished %s()' % (func_name))
            return result

        slot_wrapper.func_name = func_name
        return slot_wrapper
    return pyqtSlotWrapper


def fastslot_(*types):
    'wrapper around pyqtslot decorator'

    # Wrap wihout any debugging
    def pyqtSlotWrapper(func):
        func_name = func.func_name

        @Qt.pyqtSlot(*types, name=func.func_name)
        def slot_wrapper(self, *args, **kwargs):
            return func(self, *args, **kwargs)
        slot_wrapper.func_name = func_name
        return slot_wrapper
    return pyqtSlotWrapper

slot_ = dbgslot_ if DEBUG else fastslot_
#/SLOT DECORATOR
#---------------


# BLOCKING DECORATOR
# TODO: This decorator has to be specific to either front or back. Is there a
# way to make it more general?
def backblocking(func):
    #printDBG('[@guitools] Wrapping %r with backblocking' % func.func_name)

    def block_wrapper(back, *args, **kwargs):
        #print('[guitools] BLOCKING')
        wasBlocked_ = back.front.blockSignals(True)
        try:
            result = func(back, *args, **kwargs)
        except Exception as ex:
            back.front.blockSignals(wasBlocked_)
            print('Block wrapper caugt exception in %r' % func.func_name)
            print('back = %r' % back)
            print('*args = %r' % (args,))
            print('**kwargs = %r' % (kwargs,))
            print('ex = %r' % ex)
            back.user_info('Error in blocking ex=%r' % ex)
            raise
        back.front.blockSignals(wasBlocked_)
        #print('[guitools] UNBLOCKING')
        return result
    block_wrapper.func_name = func.func_name
    return block_wrapper


def frontblocking(func):
    # HACK: blocking2 is specific to fron
    #printDBG('[@guitools] Wrapping %r with frontblocking' % func.func_name)

    def block_wrapper(front, *args, **kwargs):
        #print('[guitools] BLOCKING')
        #wasBlocked = self.blockSignals(True)
        wasBlocked_ = front.blockSignals(True)
        try:
            result = func(front, *args, **kwargs)
        except Exception as ex:
            front.blockSignals(wasBlocked_)
            print('Block wrapper caugt exception in %r' % func.func_name)
            print('front = %r' % front)
            print('*args = %r' % (args,))
            print('**kwargs = %r' % (kwargs,))
            print('ex = %r' % ex)
            front.user_info('Error in blocking ex=%r' % ex)
            raise
        front.blockSignals(wasBlocked_)
        #print('[guitools] UNBLOCKING')
        return result
    block_wrapper.func_name = func.func_name
    return block_wrapper


# DRAWING DECORATOR
def drawing(func):
    'Wraps a class function and draws windows on completion'
    #printDBG('[@guitools] Wrapping %r with drawing' % func.func_name)

    def drawing_wrapper(self, *args, **kwargs):
        #print('[guitools] DRAWING')
        result = func(self, *args, **kwargs)
        #print('[guitools] DONE DRAWING')
        if kwargs.get('dodraw', True) or DISABLE_NODRAW:
            df2.draw()
        return result
    drawing_wrapper.func_name = func.func_name
    return drawing_wrapper


def select_orientation():
    #from matplotlib.backend_bases import mplDeprecation
    print('[*guitools] Define an orientation angle by clicking two points')
    try:
        # Compute an angle from user interaction
        sys.stdout.flush()
        fig = df2.gcf()
        oldcbid, oldcbfn = df2.disconnect_callback(fig, 'button_press_event')
        #with warnings.catch_warnings():
            #warnings.filterwarnings("ignore", category=mplDeprecation)
        pts = np.array(fig.ginput(2))
        #print('[*guitools] ginput(2) = %r' % pts)
        # Get reference point to origin
        refpt = pts[1] - pts[0]
        #theta = np.math.atan2(refpt[1], refpt[0])
        theta = np.math.atan2(refpt[1], refpt[0])
        print('The angle in radians is: %r' % theta)
        df2.connect_callback(fig, 'button_press_event', oldcbfn)
        return theta
    except Exception as ex:
        print('Annotate Orientation Failed %r' % ex)
        return None


def select_roi():
    #from matplotlib.backend_bases import mplDeprecation
    print('[*guitools] Define a Rectanglular ROI by clicking two points.')
    try:
        sys.stdout.flush()
        fig = df2.gcf()
        # Disconnect any other button_press events
        oldcbid, oldcbfn = df2.disconnect_callback(fig, 'button_press_event')
        #with warnings.catch_warnings():
            #warnings.filterwarnings("ignore", category=mplDeprecation)
        pts = fig.ginput(2)
        print('[*guitools] ginput(2) = %r' % (pts,))
        [(x1, y1), (x2, y2)] = pts
        xm = min(x1, x2)
        xM = max(x1, x2)
        ym = min(y1, y2)
        yM = max(y1, y2)
        xywh = map(int, map(round, (xm, ym, xM - xm, yM - ym)))
        roi = np.array(xywh, dtype=np.int32)
        # Reconnect the old button press events
        df2.connect_callback(fig, 'button_press_event', oldcbfn)
        print('[*guitools] roi = %r ' % (roi,))
        return roi
    except Exception as ex:
        print('[*guitools] ROI selection Failed:\n%r' % (ex,))
        return None


def _addOptions(msgBox, options):
    #msgBox.addButton(Qt.QMessageBox.Close)
    for opt in options:
        role = QtGui.QMessageBox.ApplyRole
        msgBox.addButton(QtGui.QPushButton(opt), role)


def _cacheReply(msgBox):
    dontPrompt = QtGui.QCheckBox('dont ask me again', parent=msgBox)
    dontPrompt.blockSignals(True)
    msgBox.addButton(dontPrompt, Qt.QMessageBox.ActionRole)
    return dontPrompt


def _newMsgBox(msg='', title='', parent=None, options=None, cache_reply=False):
    msgBox = QtGui.QMessageBox(parent)
    #msgBox.setAttribute(QtCore.Qt.WA_DeleteOnClose)
    #std_buts = Qt.QMessageBox.Close
    #std_buts = Qt.QMessageBox.NoButton
    std_buts = Qt.QMessageBox.Cancel
    msgBox.setStandardButtons(std_buts)
    msgBox.setWindowTitle(title)
    msgBox.setText(msg)
    msgBox.setModal(parent is not None)
    return msgBox


def msgbox(msg, title='msgbox'):
    'Make a non modal critical Qt.QMessageBox.'
    msgBox = Qt.QMessageBox(None)
    msgBox.setAttribute(QtCore.Qt.WA_DeleteOnClose)
    msgBox.setStandardButtons(Qt.QMessageBox.Ok)
    msgBox.setWindowTitle(title)
    msgBox.setText(msg)
    msgBox.setModal(False)
    msgBox.open(msgBox.close)
    msgBox.show()
    return msgBox


def user_input(parent, msg, title='input dialog'):
    reply, ok = QtGui.QInputDialog.getText(parent, title, msg)
    if not ok:
        return None
    return str(reply)


def user_info(parent, msg, title='info'):
    msgBox = _newMsgBox(msg, title, parent)
    msgBox.setAttribute(QtCore.Qt.WA_DeleteOnClose)
    msgBox.setStandardButtons(Qt.QMessageBox.Ok)
    msgBox.setModal(False)
    msgBox.open(msgBox.close)
    msgBox.show()


def user_option(parent, msg, title='options', options=['No', 'Yes'], use_cache=False):
    'Prompts user with several options with ability to save decision'
    print('[*guitools] user_option:\n %r: %s' + title + ': ' + msg)
    # Recall decision
    cache_id = helpers.hashstr(title + msg)
    if use_cache:
        reply = io.global_cache_read(cache_id, default=None)
        if reply is not None:
            return reply
    # Create message box
    msgBox = _newMsgBox(msg, title, parent)
    _addOptions(msgBox, options)
    if use_cache:
        dontPrompt = _cacheReply(msgBox)
    # Wait for output
    optx = msgBox.exec_()
    if optx == Qt.QMessageBox.Cancel:
        return None
    try:
        reply = options[optx]
    except Exception as ex:
        print('[*guitools] USER OPTION EXCEPTION !')
        print('[*guitools] optx = %r' % optx)
        print('[*guitools] options = %r' % options)
        print('[*guitools] ex = %r' % ex)
        raise
    # Remember decision
    if use_cache and dontPrompt.isChecked():
        io.global_cache_write(cache_id, reply)
    del msgBox
    return reply


def user_question(msg):
    msgBox = Qt.QMessageBox.question(None, '', 'lovely day?')
    return msgBox


def getQtImageNameFilter():
    imgNamePat = ' '.join(['*' + ext for ext in helpers.IMG_EXTENSIONS])
    imgNameFilter = 'Images (%s)' % (imgNamePat)
    return imgNameFilter


def select_images(caption='Select images:', directory=None):
    name_filter = getQtImageNameFilter()
    return select_files(caption, directory, name_filter)


def select_files(caption='Select Files:', directory=None, name_filter=None):
    'Selects one or more files from disk using a qt dialog'
    print(caption)
    if directory is None:
        directory = io.global_cache_read('select_directory')
    qdlg = PyQt4.Qt.QFileDialog()
    qfile_list = qdlg.getOpenFileNames(caption=caption, directory=directory, filter=name_filter)
    file_list = map(str, qfile_list)
    print('Selected %d files' % len(file_list))
    io.global_cache_write('select_directory', directory)
    return file_list


def select_directory(caption='Select Directory', directory=None):
    print(caption)
    if directory is None:
        directory = io.global_cache_read('select_directory')
    qdlg = PyQt4.Qt.QFileDialog()
    qopt = PyQt4.Qt.QFileDialog.ShowDirsOnly
    qdlg_kwargs = dict(caption=caption, options=qopt, directory=directory)
    dpath = str(qdlg.getExistingDirectory(**qdlg_kwargs))
    print('Selected Directory: %r' % dpath)
    io.global_cache_write('select_directory', split(dpath)[0])
    return dpath


def show_open_db_dlg(parent=None):
    # OLD
    from _frontend import OpenDatabaseDialog
    if not '-nc' in sys.argv and not '--nocache' in sys.argv:
        db_dir = io.global_cache_read('db_dir')
        if db_dir == '.':
            db_dir = None
    print('[*guitools] cached db_dir=%r' % db_dir)
    if parent is None:
        parent = PyQt4.QtGui.QDialog()
    opendb_ui = OpenDatabaseDialog.Ui_Dialog()
    opendb_ui.setupUi(parent)
    #opendb_ui.new_db_but.clicked.connect(create_new_database)
    #opendb_ui.open_db_but.clicked.connect(open_old_database)
    parent.show()
    return opendb_ui, parent


def init_qtapp():
    global IS_INIT
    app = Qt.QCoreApplication.instance()
    is_root = app is None
    if is_root:  # if not in qtconsole
        print('[*guitools] Initializing QApplication')
        app = Qt.QApplication(sys.argv)
    try:
        __IPYTHON__
        is_root = False
    # You are not root if you are in IPYTHON
    except NameError:
        pass
    IS_INIT = True
    return app, is_root


def exit_application():
    print('[*guitools] exiting application')
    QtGui.qApp.quit()


def run_main_loop(app, is_root=True, back=None, **kwargs):
    if back is not None:
        print('[*guitools] setting active window')
        app.setActiveWindow(back.front)
        back.timer = ping_python_interpreter(**kwargs)
    if is_root:
        exec_core_app_loop(app)
        #exec_core_event_loop(app)
    else:
        print('[*guitools] using roots main loop')


def exec_core_event_loop(app):
    # This works but does not allow IPython injection
    print('[*guitools] running core application loop.')
    try:
        from IPython.lib.inputhook import enable_qt4
        enable_qt4()
        from IPython.lib.guisupport import start_event_loop_qt4
        print('Starting ipython qt4 hook')
        start_event_loop_qt4(app)
    except ImportError:
        pass
    app.exec_()


def exec_core_app_loop(app):
    # This works but does not allow IPython injection
    print('[*guitools] running core application loop.')
    app.exec_()
    #sys.exit(app.exec_())


def ping_python_interpreter(frequency=4200):  # 4200):
    'Create a QTimer which lets the python catch ctrl+c'
    timer = Qt.QTimer()
    timer.timeout.connect(lambda: None)
    timer.start(frequency)
    return timer


def make_dummy_main_window():
    class DummyBackend(Qt.QObject):
        def __init__(self):
            super(DummyBackend,  self).__init__()
            self.front = PyQt4.Qt.QMainWindow()
            self.front.setWindowTitle('Dummy Main Window')
            self.front.show()
    back = DummyBackend()
    return back


def popup_menu(widget, opt2_callback):
    def popup_slot(pos):
        print(pos)
        menu = QtGui.QMenu()
        actions = [menu.addAction(opt, func) for opt, func in
                   iter(opt2_callback)]
        #pos=QtGui.QCursor.pos()
        selection = menu.exec_(widget.mapToGlobal(pos))
        return selection, actions
    return popup_slot


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    print('__main__ = gui.py')
    app, is_root = init_qtapp()
    back = make_dummy_main_window()
    front = back.front
    run_main_loop(app, is_root, back)
