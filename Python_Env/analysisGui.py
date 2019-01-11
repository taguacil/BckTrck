"""
 =============================================================================
 Title       : Gui for analysis
 Project     : Simulation environment for BckTrk app
 File        : analysisGui.py
 -----------------------------------------------------------------------------

   Description :

   This file is responsible for gui plotting of analysis

   References :

   -
 -----------------------------------------------------------------------------
 Revisions   :
   Date         Version  Name      Description
   25-Sep-2018  1.0      Taimir    File created
 =============================================================================

"""
import sys

from PyQt5 import QtCore
from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import Helper_functions.analysisCompute


def create_main_frame(self):
    self.main_frame = QtWidgets.QWidget()

    # Create the mpl Figure and FigCanvas objects.
    # 5x4 inches, 100 dots-per-inch
    #
    self.dpi = 100
    self.fig = Figure((12.0, 8.0), dpi=self.dpi)
    self.canvas = FigureCanvas(self.fig)
    self.canvas.setParent(self.main_frame)

    # Bind the 'pick' event for clicking on one of the bars
    #
    self.canvas.mpl_connect('pick_event', self.on_pick)

    # Create the navigation toolbar, tied to the canvas
    #
    self.mpl_toolbar = NavigationToolbar(self.canvas, self.main_frame)

    # Other GUI controls
    #
    self.textbox = QtWidgets.QLineEdit()
    self.textbox.setMinimumWidth(200)
    self.textbox.editingFinished.connect(self.on_draw)

    self.draw_button = QtWidgets.QPushButton("&Draw")
    self.draw_button.clicked.connect(self.on_draw)

    self.grid_cb = QtWidgets.QCheckBox("Show &Grid")
    self.grid_cb.setChecked(False)
    self.grid_cb.stateChanged.connect(self.on_draw)

    self.log_option = QtWidgets.QCheckBox("Log colormap")
    self.log_option.setChecked(False)
    self.log_option.stateChanged.connect(self.on_draw)

    self.SNR_plots = QtWidgets.QCheckBox("Plot SNR")
    self.SNR_plots.setChecked(False)
    self.SNR_plots.stateChanged.connect(self.on_draw)

    #
    # Layout with box sizers
    #
    hbox = QtWidgets.QHBoxLayout()

    for w in [self.textbox, self.draw_button, self.grid_cb, self.log_option, self.SNR_plots]:
        hbox.addWidget(w)
        hbox.setAlignment(w, QtCore.Qt.AlignVCenter)

    vbox = QtWidgets.QVBoxLayout()
    vbox.addWidget(self.canvas)
    vbox.addWidget(self.mpl_toolbar)
    vbox.addLayout(hbox)

    self.main_frame.setLayout(vbox)
    self.setCentralWidget(self.main_frame)


def create_menu(self):
    self.file_menu = self.menuBar().addMenu("&File")

    load_file_action = self.create_action("&Save plot",
                                          shortcut="Ctrl+S", slot=self.save_plot,
                                          tip="Save the plot")
    quit_action = self.create_action("&Quit", slot=self.close,
                                     shortcut="Ctrl+Q", tip="Close the application")

    self.add_actions(self.file_menu,
                     (load_file_action, None, quit_action))

    self.help_menu = self.menuBar().addMenu("&Help")
    about_action = self.create_action("&About",
                                      shortcut='F1', slot=self.on_about,
                                      tip='About the analysis')

    self.add_actions(self.help_menu, (about_action,))


def create_status_bar(self):
    self.status_text = QtWidgets.QLabel("Scan analysis plotting")
    self.statusBar().addWidget(self.status_text, 1)


class AppForm(QtWidgets.QMainWindow):
    def __init__(self, CScanAnalysis, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)
        self.setWindowTitle('PyQt with matplotlib')

        self.axes = {}

        create_menu(self)
        create_main_frame(self)
        create_status_bar(self)

        self.table_lin = CScanAnalysis.get_panda_table()
        self.table_lin = self.table_lin[self.table_lin.SNR != 0]
        self.table_log = self.table_lin.copy()
        self.table_log.MSE = np.log10(self.table_log.MSE)
        self.sampling_ratios_values, self.path_lengths_values, \
        self.learning_rates_values, self.noise_levels_values = \
            CScanAnalysis.get_slices_values()

        self.textbox.setText(str(self.sampling_ratios_values[0]) + ' ' + str(self.path_lengths_values[0]) + ' ' +
                             str(self.learning_rates_values[0]) + ' ' + str(self.noise_levels_values[0]))

        self.individualPlot()
        self.on_draw()

    def individualPlot(self):
        font = {'family': 'normal',
                'size': 14}

        matplotlib.rc('font', **font)
        plt.scatter(
            self.table_lin[(self.table_lin.LearningRate == 0.1) & (self.table_lin.Noise == 20)]['SamplingRatio'],
            self.table_lin[(self.table_lin.LearningRate == 0.1) & (self.table_lin.Noise == 20)]['PathLengths'],
            c=self.table_lin[(self.table_lin.LearningRate == 0.1) & (self.table_lin.Noise == 20)]['MSE'],
            cmap='rainbow_r'
            )
        plt.yscale('log')
        plt.xlabel('Sampling ratio')
        plt.ylabel('Path length')
        cb = plt.colorbar()
        cb.set_label('MSE (meters)')
        plt.rcParams["figure.figsize"] = [10, 10]
        plt.show()

    def save_plot(self):
        file_choices = "PNG (*.png)|*.png"

        path = str(QtWidgets.QFileDialog.getSaveFileName(self, 'Save file', '', file_choices))
        if path:
            self.canvas.print_figure(path, dpi=self.dpi)
            self.statusBar().showMessage('Saved to %s' % path, 2000)

    def on_about(self):
        msg = """ Analysis using PyQt with matplotlib:

         * Use the matplotlib navigation bar
         * Add values to the text box and press Enter (or click "Draw")
         * Show or hide the grid
         * Drag the slider to modify the width of the bars
         * Save the plot to a file using the File menu
         * Click on a bar to receive an informative message
        """
        QtWidgets.QMessageBox.about(self, "About the analysis", msg.strip())

    def on_pick(self, event):
        # The event received here is of the type
        # matplotlib.backend_bases.PickEvent
        #
        # It carries lots of information, of which we're using
        # only a small amount here.
        #
        box_points = event.artist.get_bbox().get_points()
        msg = "You've clicked on a bar with coords:\n %s" % box_points

        QtWidgets.QMessageBox.information(self, "Click!", msg)

    def on_draw(self):
        """ Redraws the figure
        """
        stringVar = str(self.textbox.text())
        try:
            self.data = map(float, stringVar.split())
            sliceTuple = list(self.data)
        except (ValueError, UnboundLocalError):
            print("Error in input value, displaying default")

        sampling_ratios_slice = self.sampling_ratios_values[self.sampling_ratios_values == sliceTuple[0]]
        if sampling_ratios_slice.size == 0:
            sampling_ratios_slice = self.sampling_ratios_values[0]
        else:
            sampling_ratios_slice = sampling_ratios_slice[0]

        path_lengths_slice = self.path_lengths_values[self.path_lengths_values == sliceTuple[1]]
        if path_lengths_slice.size == 0:
            path_lengths_slice = self.path_lengths_values[0]
        else:
            path_lengths_slice = path_lengths_slice[0]

        learning_rates_slice = self.learning_rates_values[self.learning_rates_values == sliceTuple[2]]
        if learning_rates_slice.size == 0:
            learning_rates_slice = self.learning_rates_values[0]
        else:
            learning_rates_slice = learning_rates_slice[0]

        noise_levels_slice = self.noise_levels_values[self.noise_levels_values == sliceTuple[3]]
        if noise_levels_slice.size == 0:
            noise_levels_slice = self.noise_levels_values[0]
        else:
            noise_levels_slice = noise_levels_slice[0]

        # clear the axes and redraw the plot anew
        self.fig.clear()
        self.axes = self.fig.subplots(nrows=3, ncols=3)
        bgrid = self.grid_cb.isChecked()

        if self.log_option.isChecked():
            table = self.table_log
        else:
            table = self.table_lin

        if self.SNR_plots.isChecked():
            column_iden = 'SNR'
        else:
            column_iden = 'MSE'

        table[(table.LearningRate == learning_rates_slice) &
              (table.Noise == noise_levels_slice)]. \
            plot.scatter('SamplingRatio', 'PathLengths', c=column_iden, colormap='rainbow_r', ax=self.axes[0, 0],
                         title='Lr <%.3f>, Noise <%.4f>' % (learning_rates_slice, noise_levels_slice),
                         logy=True, grid=bgrid)

        table[(table.PathLengths == path_lengths_slice) &
              (table.Noise == noise_levels_slice)]. \
            plot.scatter('SamplingRatio', 'LearningRate', c=column_iden, colormap='rainbow_r', ax=self.axes[0, 1],
                         title='Pl <%d>, Noise <%.4f>' % (path_lengths_slice, noise_levels_slice),
                         logy=True, grid=bgrid)

        table[(table.LearningRate == learning_rates_slice) &
              (table.PathLengths == path_lengths_slice)]. \
            plot.scatter('SamplingRatio', 'Noise', c=column_iden, colormap='rainbow_r', ax=self.axes[0, 2],
                         title='Pl <%d>, Lr <%.3f>' % (path_lengths_slice, learning_rates_slice),
                         logy=True, grid=bgrid)

        table[(table.SamplingRatio == sampling_ratios_slice) &
              (table.Noise == noise_levels_slice)]. \
            plot.scatter('PathLengths', 'LearningRate', c=column_iden, colormap='rainbow_r', ax=self.axes[1, 0],
                         title='Sr <%.3f>, Noise <%.4f>' % (sampling_ratios_slice, noise_levels_slice),
                         logy=True, grid=bgrid)

        table[(table.LearningRate == learning_rates_slice) &
              (table.SamplingRatio == sampling_ratios_slice)]. \
            plot.scatter('PathLengths', 'Noise', c=column_iden, colormap='rainbow_r', ax=self.axes[1, 1],
                         title='Sr <%.3f>, Lr <%.3f>' % (sampling_ratios_slice, learning_rates_slice),
                         logy=True, grid=bgrid)

        table[(table.SamplingRatio == sampling_ratios_slice) &
              (table.PathLengths == path_lengths_slice)]. \
            plot.scatter('LearningRate', 'Noise', c=column_iden, colormap='rainbow_r', ax=self.axes[1, 2],
                         title='Lr <%.3f>, Noise <%.4f>' % (learning_rates_slice, noise_levels_slice),
                         logy=True, grid=bgrid)

        table[(table.LearningRate == learning_rates_slice) &
              (table.SamplingRatio == sampling_ratios_slice)]. \
            plot.scatter('PathLengths', 'Noise', c='L1Lat', colormap='rainbow_r', ax=self.axes[2, 0],
                         title='L1 norm for latitude', logy=True, grid=bgrid)

        table[(table.LearningRate == learning_rates_slice) &
              (table.SamplingRatio == sampling_ratios_slice)]. \
            plot.scatter('PathLengths', 'Noise', c='L1Lon', colormap='rainbow_r', ax=self.axes[2, 1],
                         title='L1 norm for longitude', logy=True, grid=bgrid)

        self.fig.tight_layout()
        self.canvas.draw()

    @staticmethod
    def add_actions(target, actions):
        for action in actions:
            if action is None:
                target.addSeparator()
            else:
                target.addAction(action)

    def create_action(self, text, slot=None, shortcut=None,
                      icon=None, tip=None, checkable=False,
                      signal="triggered()"):
        action = QtWidgets.QAction(text, self)
        if icon is not None:
            action.setIcon(QtWidgets.QIcon(":/%s.png" % icon))
        if shortcut is not None:
            action.setShortcut(shortcut)
        if tip is not None:
            action.setToolTip(tip)
            action.setStatusTip(tip)
        if slot is not None:
            action.triggered.connect(slot)
        if checkable:
            action.setCheckable(True)
        return action


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    analyzeObj = Helper_functions.analysisCompute.CScanAnalysis(sys.argv)
    analyzeObj.analyze()

    form = AppForm(analyzeObj)
    form.show()

    app.exec_()
