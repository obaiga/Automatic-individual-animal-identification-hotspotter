# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\jon.crall\code\hotspotter\setup_helpers\../hotspotter/front\MainSkel.ui'
#
# Created: Sat May 25 20:22:37 2013
#      by: PyQt4 UI code generator 4.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_mainSkel(object):
    def setupUi(self, mainSkel):
        mainSkel.setObjectName(_fromUtf8("mainSkel"))
        mainSkel.resize(927, 573)
        self.centralwidget = QtGui.QWidget(mainSkel)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.gridLayout_2 = QtGui.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName(_fromUtf8("gridLayout_2"))
        self.root_hlayout = QtGui.QHBoxLayout()
        self.root_hlayout.setObjectName(_fromUtf8("root_hlayout"))
        self.tablesTabWidget = QtGui.QTabWidget(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tablesTabWidget.sizePolicy().hasHeightForWidth())
        self.tablesTabWidget.setSizePolicy(sizePolicy)
        self.tablesTabWidget.setObjectName(_fromUtf8("tablesTabWidget"))
        self.image_view = QtGui.QWidget()
        self.image_view.setMinimumSize(QtCore.QSize(445, 0))
        self.image_view.setObjectName(_fromUtf8("image_view"))
        self.gridLayout_3 = QtGui.QGridLayout(self.image_view)
        self.gridLayout_3.setObjectName(_fromUtf8("gridLayout_3"))
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.image_TBL = QtGui.QTableWidget(self.image_view)
        self.image_TBL.setObjectName(_fromUtf8("image_TBL"))
        self.image_TBL.setColumnCount(0)
        self.image_TBL.setRowCount(0)
        self.verticalLayout.addWidget(self.image_TBL)
        self.gridLayout_3.addLayout(self.verticalLayout, 0, 0, 1, 1)
        self.tablesTabWidget.addTab(self.image_view, _fromUtf8(""))
        self.chip_view = QtGui.QWidget()
        self.chip_view.setObjectName(_fromUtf8("chip_view"))
        self.gridLayout_4 = QtGui.QGridLayout(self.chip_view)
        self.gridLayout_4.setObjectName(_fromUtf8("gridLayout_4"))
        self.verticalLayout_3 = QtGui.QVBoxLayout()
        self.verticalLayout_3.setObjectName(_fromUtf8("verticalLayout_3"))
        self.chip_TBL = QtGui.QTableWidget(self.chip_view)
        self.chip_TBL.setObjectName(_fromUtf8("chip_TBL"))
        self.chip_TBL.setColumnCount(0)
        self.chip_TBL.setRowCount(0)
        self.verticalLayout_3.addWidget(self.chip_TBL)
        self.gridLayout_4.addLayout(self.verticalLayout_3, 0, 0, 1, 1)
        self.tablesTabWidget.addTab(self.chip_view, _fromUtf8(""))
        self.result_view = QtGui.QWidget()
        self.result_view.setObjectName(_fromUtf8("result_view"))
        self.verticalLayout_5 = QtGui.QVBoxLayout(self.result_view)
        self.verticalLayout_5.setObjectName(_fromUtf8("verticalLayout_5"))
        self.verticalLayout_4 = QtGui.QVBoxLayout()
        self.verticalLayout_4.setObjectName(_fromUtf8("verticalLayout_4"))
        self.res_TBL = QtGui.QTableWidget(self.result_view)
        self.res_TBL.setObjectName(_fromUtf8("res_TBL"))
        self.res_TBL.setColumnCount(0)
        self.res_TBL.setRowCount(0)
        self.verticalLayout_4.addWidget(self.res_TBL)
        self.verticalLayout_5.addLayout(self.verticalLayout_4)
        self.tablesTabWidget.addTab(self.result_view, _fromUtf8(""))
        self.root_hlayout.addWidget(self.tablesTabWidget)
        self.gridLayout_2.addLayout(self.root_hlayout, 1, 1, 1, 1)
        self.status_HLayout = QtGui.QHBoxLayout()
        self.status_HLayout.setObjectName(_fromUtf8("status_HLayout"))
        self.label_2 = QtGui.QLabel(self.centralwidget)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.status_HLayout.addWidget(self.label_2)
        self.fignumSPIN = QtGui.QSpinBox(self.centralwidget)
        self.fignumSPIN.setObjectName(_fromUtf8("fignumSPIN"))
        self.status_HLayout.addWidget(self.fignumSPIN)
        self.state_LBL = QtGui.QLabel(self.centralwidget)
        self.state_LBL.setObjectName(_fromUtf8("state_LBL"))
        self.status_HLayout.addWidget(self.state_LBL)
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.status_HLayout.addItem(spacerItem)
        spacerItem1 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.status_HLayout.addItem(spacerItem1)
        self.label_3 = QtGui.QLabel(self.centralwidget)
        self.label_3.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.status_HLayout.addWidget(self.label_3)
        self.sel_cid_SPIN = QtGui.QSpinBox(self.centralwidget)
        self.sel_cid_SPIN.setEnabled(False)
        self.sel_cid_SPIN.setObjectName(_fromUtf8("sel_cid_SPIN"))
        self.status_HLayout.addWidget(self.sel_cid_SPIN)
        self.label = QtGui.QLabel(self.centralwidget)
        self.label.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label.setObjectName(_fromUtf8("label"))
        self.status_HLayout.addWidget(self.label)
        self.sel_gid_SPIN = QtGui.QSpinBox(self.centralwidget)
        self.sel_gid_SPIN.setEnabled(False)
        self.sel_gid_SPIN.setObjectName(_fromUtf8("sel_gid_SPIN"))
        self.status_HLayout.addWidget(self.sel_gid_SPIN)
        self.gridLayout_2.addLayout(self.status_HLayout, 2, 1, 1, 1)
        mainSkel.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(mainSkel)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 927, 21))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        self.menuFile = QtGui.QMenu(self.menubar)
        self.menuFile.setObjectName(_fromUtf8("menuFile"))
        self.menuView = QtGui.QMenu(self.menubar)
        self.menuView.setObjectName(_fromUtf8("menuView"))
        self.menuOptions = QtGui.QMenu(self.menubar)
        self.menuOptions.setObjectName(_fromUtf8("menuOptions"))
        self.menuHelp = QtGui.QMenu(self.menubar)
        self.menuHelp.setObjectName(_fromUtf8("menuHelp"))
        self.menuActions = QtGui.QMenu(self.menubar)
        self.menuActions.setObjectName(_fromUtf8("menuActions"))
        self.menuExperiments = QtGui.QMenu(self.menubar)
        self.menuExperiments.setObjectName(_fromUtf8("menuExperiments"))
        mainSkel.setMenuBar(self.menubar)
        self.actionOpen_Database = QtGui.QAction(mainSkel)
        self.actionOpen_Database.setObjectName(_fromUtf8("actionOpen_Database"))
        self.actionSave_Database = QtGui.QAction(mainSkel)
        self.actionSave_Database.setObjectName(_fromUtf8("actionSave_Database"))
        self.actionImport_Images = QtGui.QAction(mainSkel)
        self.actionImport_Images.setObjectName(_fromUtf8("actionImport_Images"))
        self.actionOpen_Data_Directory = QtGui.QAction(mainSkel)
        self.actionOpen_Data_Directory.setObjectName(_fromUtf8("actionOpen_Data_Directory"))
        self.actionOpen_Source_Directory = QtGui.QAction(mainSkel)
        self.actionOpen_Source_Directory.setObjectName(_fromUtf8("actionOpen_Source_Directory"))
        self.actionTogEll = QtGui.QAction(mainSkel)
        self.actionTogEll.setObjectName(_fromUtf8("actionTogEll"))
        self.actionUndockDisplay = QtGui.QAction(mainSkel)
        self.actionUndockDisplay.setObjectName(_fromUtf8("actionUndockDisplay"))
        self.actionTogPlt = QtGui.QAction(mainSkel)
        self.actionTogPlt.setObjectName(_fromUtf8("actionTogPlt"))
        self.actionHelpCMD = QtGui.QAction(mainSkel)
        self.actionHelpCMD.setObjectName(_fromUtf8("actionHelpCMD"))
        self.actionHelpGUI = QtGui.QAction(mainSkel)
        self.actionHelpGUI.setObjectName(_fromUtf8("actionHelpGUI"))
        self.actionHelpTroubles = QtGui.QAction(mainSkel)
        self.actionHelpTroubles.setObjectName(_fromUtf8("actionHelpTroubles"))
        self.actionHelpWorkflow = QtGui.QAction(mainSkel)
        self.actionHelpWorkflow.setObjectName(_fromUtf8("actionHelpWorkflow"))
        self.actionTogPts = QtGui.QAction(mainSkel)
        self.actionTogPts.setObjectName(_fromUtf8("actionTogPts"))
        self.actionAdd_ROI = QtGui.QAction(mainSkel)
        self.actionAdd_ROI.setObjectName(_fromUtf8("actionAdd_ROI"))
        self.actionReselect_ROI = QtGui.QAction(mainSkel)
        self.actionReselect_ROI.setObjectName(_fromUtf8("actionReselect_ROI"))
        self.actionNext = QtGui.QAction(mainSkel)
        self.actionNext.setObjectName(_fromUtf8("actionNext"))
        self.actionRemove_Chip = QtGui.QAction(mainSkel)
        self.actionRemove_Chip.setObjectName(_fromUtf8("actionRemove_Chip"))
        self.actionQuery = QtGui.QAction(mainSkel)
        self.actionQuery.setObjectName(_fromUtf8("actionQuery"))
        self.actionPrev = QtGui.QAction(mainSkel)
        self.actionPrev.setObjectName(_fromUtf8("actionPrev"))
        self.actionWriteLogs = QtGui.QAction(mainSkel)
        self.actionWriteLogs.setObjectName(_fromUtf8("actionWriteLogs"))
        self.actionOpen_Internal_Directory = QtGui.QAction(mainSkel)
        self.actionOpen_Internal_Directory.setObjectName(_fromUtf8("actionOpen_Internal_Directory"))
        self.actionPreferences = QtGui.QAction(mainSkel)
        self.actionPreferences.setObjectName(_fromUtf8("actionPreferences"))
        self.actionQuit = QtGui.QAction(mainSkel)
        self.actionQuit.setObjectName(_fromUtf8("actionQuit"))
        self.actionConvertImage2Chip = QtGui.QAction(mainSkel)
        self.actionConvertImage2Chip.setObjectName(_fromUtf8("actionConvertImage2Chip"))
        self.actionBatch_Change_Name = QtGui.QAction(mainSkel)
        self.actionBatch_Change_Name.setObjectName(_fromUtf8("actionBatch_Change_Name"))
        self.actionReselect_Orientation = QtGui.QAction(mainSkel)
        self.actionReselect_Orientation.setObjectName(_fromUtf8("actionReselect_Orientation"))
        self.actionAdd_Metadata_Property = QtGui.QAction(mainSkel)
        self.actionAdd_Metadata_Property.setObjectName(_fromUtf8("actionAdd_Metadata_Property"))
        self.actionAssign_Matches_Above_Threshold = QtGui.QAction(mainSkel)
        self.actionAssign_Matches_Above_Threshold.setObjectName(_fromUtf8("actionAssign_Matches_Above_Threshold"))
        self.actionMatching_Experiment = QtGui.QAction(mainSkel)
        self.actionMatching_Experiment.setObjectName(_fromUtf8("actionMatching_Experiment"))
        self.actionName_Consistency_Experiment = QtGui.QAction(mainSkel)
        self.actionName_Consistency_Experiment.setObjectName(_fromUtf8("actionName_Consistency_Experiment"))
        self.actionIncrease_ROI_Size = QtGui.QAction(mainSkel)
        self.actionIncrease_ROI_Size.setObjectName(_fromUtf8("actionIncrease_ROI_Size"))
        self.menuFile.addAction(self.actionOpen_Database)
        self.menuFile.addAction(self.actionSave_Database)
        self.menuFile.addAction(self.actionImport_Images)
        self.menuFile.addAction(self.actionQuit)
        self.menuView.addAction(self.actionConvertImage2Chip)
        self.menuView.addAction(self.actionBatch_Change_Name)
        self.menuView.addSeparator()
        self.menuView.addAction(self.actionAdd_Metadata_Property)
        self.menuView.addAction(self.actionAssign_Matches_Above_Threshold)
        self.menuView.addAction(self.actionIncrease_ROI_Size)
        self.menuView.addSeparator()
        self.menuView.addAction(self.actionOpen_Data_Directory)
        self.menuView.addAction(self.actionOpen_Source_Directory)
        self.menuView.addAction(self.actionOpen_Internal_Directory)
        self.menuOptions.addAction(self.actionTogEll)
        self.menuOptions.addAction(self.actionTogPts)
        self.menuOptions.addSeparator()
        self.menuOptions.addAction(self.actionTogPlt)
        self.menuOptions.addSeparator()
        self.menuOptions.addAction(self.actionPreferences)
        self.menuHelp.addAction(self.actionHelpWorkflow)
        self.menuHelp.addAction(self.actionHelpTroubles)
        self.menuHelp.addSeparator()
        self.menuHelp.addAction(self.actionHelpCMD)
        self.menuHelp.addSeparator()
        self.menuHelp.addAction(self.actionWriteLogs)
        self.menuActions.addAction(self.actionQuery)
        self.menuActions.addSeparator()
        self.menuActions.addAction(self.actionAdd_ROI)
        self.menuActions.addAction(self.actionReselect_ROI)
        self.menuActions.addAction(self.actionReselect_Orientation)
        self.menuActions.addAction(self.actionRemove_Chip)
        self.menuActions.addSeparator()
        self.menuActions.addAction(self.actionNext)
        self.menuExperiments.addAction(self.actionMatching_Experiment)
        self.menuExperiments.addAction(self.actionName_Consistency_Experiment)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuActions.menuAction())
        self.menubar.addAction(self.menuOptions.menuAction())
        self.menubar.addAction(self.menuView.menuAction())
        self.menubar.addAction(self.menuExperiments.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())

        self.retranslateUi(mainSkel)
        self.tablesTabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(mainSkel)

    def retranslateUi(self, mainSkel):
        mainSkel.setWindowTitle(_translate("mainSkel", "HotSpotter", None))
        self.image_TBL.setSortingEnabled(True)
        self.tablesTabWidget.setTabText(self.tablesTabWidget.indexOf(self.image_view), _translate("mainSkel", "Image Table", None))
        self.chip_TBL.setSortingEnabled(True)
        self.tablesTabWidget.setTabText(self.tablesTabWidget.indexOf(self.chip_view), _translate("mainSkel", "Chip Table", None))
        self.res_TBL.setSortingEnabled(True)
        self.tablesTabWidget.setTabText(self.tablesTabWidget.indexOf(self.result_view), _translate("mainSkel", "Results Table", None))
        self.label_2.setText(_translate("mainSkel", "fignum: ", None))
        self.state_LBL.setText(_translate("mainSkel", "State: Unloaded Gui State", None))
        self.label_3.setText(_translate("mainSkel", "Image-ID:", None))
        self.label.setText(_translate("mainSkel", "Chip-ID:", None))
        self.menuFile.setTitle(_translate("mainSkel", "File", None))
        self.menuView.setTitle(_translate("mainSkel", "Convenience", None))
        self.menuOptions.setTitle(_translate("mainSkel", "Options", None))
        self.menuHelp.setTitle(_translate("mainSkel", "Help", None))
        self.menuActions.setTitle(_translate("mainSkel", "Actions", None))
        self.menuExperiments.setTitle(_translate("mainSkel", "Experiments", None))
        self.actionOpen_Database.setText(_translate("mainSkel", "Open Database", None))
        self.actionOpen_Database.setShortcut(_translate("mainSkel", "Ctrl+O", None))
        self.actionSave_Database.setText(_translate("mainSkel", "Save Database", None))
        self.actionSave_Database.setShortcut(_translate("mainSkel", "Ctrl+S", None))
        self.actionImport_Images.setText(_translate("mainSkel", "Import Images", None))
        self.actionImport_Images.setShortcut(_translate("mainSkel", "Ctrl+I", None))
        self.actionOpen_Data_Directory.setText(_translate("mainSkel", "View Data Directory", None))
        self.actionOpen_Source_Directory.setText(_translate("mainSkel", "View Source Directory", None))
        self.actionTogEll.setText(_translate("mainSkel", "Toggle Ellipses", None))
        self.actionTogEll.setShortcut(_translate("mainSkel", "E", None))
        self.actionUndockDisplay.setText(_translate("mainSkel", "Undock Display", None))
        self.actionTogPlt.setText(_translate("mainSkel", "Toggle PlotWidget", None))
        self.actionHelpCMD.setText(_translate("mainSkel", "Command Line Help", None))
        self.actionHelpGUI.setText(_translate("mainSkel", "GUI Help", None))
        self.actionHelpTroubles.setText(_translate("mainSkel", "Troubleshooting", None))
        self.actionHelpWorkflow.setText(_translate("mainSkel", "Workflow Help", None))
        self.actionTogPts.setText(_translate("mainSkel", "Toggle Points", None))
        self.actionTogPts.setShortcut(_translate("mainSkel", "P", None))
        self.actionAdd_ROI.setText(_translate("mainSkel", "Add ROI", None))
        self.actionAdd_ROI.setShortcut(_translate("mainSkel", "A", None))
        self.actionReselect_ROI.setText(_translate("mainSkel", "Reselect ROI", None))
        self.actionReselect_ROI.setShortcut(_translate("mainSkel", "R", None))
        self.actionNext.setText(_translate("mainSkel", "Select Next", None))
        self.actionNext.setToolTip(_translate("mainSkel", "Selects the next unidentified CID or Untagged GID", None))
        self.actionNext.setShortcut(_translate("mainSkel", "N", None))
        self.actionRemove_Chip.setText(_translate("mainSkel", "Remove Chip", None))
        self.actionRemove_Chip.setShortcut(_translate("mainSkel", "Ctrl+Del", None))
        self.actionQuery.setText(_translate("mainSkel", "Query", None))
        self.actionQuery.setShortcut(_translate("mainSkel", "Q", None))
        self.actionPrev.setText(_translate("mainSkel", "Prev", None))
        self.actionWriteLogs.setText(_translate("mainSkel", "Write Logs", None))
        self.actionOpen_Internal_Directory.setText(_translate("mainSkel", "View Internal Directory", None))
        self.actionPreferences.setText(_translate("mainSkel", "Edit Preferences", None))
        self.actionPreferences.setShortcut(_translate("mainSkel", "Ctrl+P", None))
        self.actionQuit.setText(_translate("mainSkel", "Quit", None))
        self.actionConvertImage2Chip.setText(_translate("mainSkel", "Convert All Images to Chips", None))
        self.actionBatch_Change_Name.setText(_translate("mainSkel", "Batch Change Name", None))
        self.actionReselect_Orientation.setText(_translate("mainSkel", "Reselect Orientation", None))
        self.actionReselect_Orientation.setShortcut(_translate("mainSkel", "O", None))
        self.actionAdd_Metadata_Property.setText(_translate("mainSkel", "Add Metadata Property", None))
        self.actionAssign_Matches_Above_Threshold.setText(_translate("mainSkel", "Assign Matches Above Threshold", None))
        self.actionMatching_Experiment.setText(_translate("mainSkel", "Matching Experiment", None))
        self.actionName_Consistency_Experiment.setText(_translate("mainSkel", "Run Name Consistency Experiment", None))
        self.actionIncrease_ROI_Size.setText(_translate("mainSkel", "Increase all ROI Sizes", None))


if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    mainSkel = QtGui.QMainWindow()
    ui = Ui_mainSkel()
    ui.setupUi(mainSkel)
    mainSkel.show()
    sys.exit(app.exec_())

